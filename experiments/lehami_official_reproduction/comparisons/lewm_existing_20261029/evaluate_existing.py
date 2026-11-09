"""Frozen-checkpoint transfer evaluation. Writes only to this comparison directory."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np

OFF = Path('/home/weizhixin/ssd/lehami_official_reproduction_20261029')
OLD = Path('/home/weizhixin/ssd/damped_pendulum_profile_20261028')
OUT = Path(__file__).resolve().parent
DATA = OFF / 'data/pendulum_multi_v2.h5'
MANIFEST = OFF / 'LeHami/lehami/data/split_pendulum_multi_v2_episode_srnn_K80_val.15_dt2_seed42.json'
HORIZON = 70


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def prepare():
    import h5py
    import hdf5plugin  # registers the original dataset codec
    manifest = json.loads(MANIFEST.read_text())
    assert sha(DATA) == manifest['dataset_sha256']
    ids = manifest['test_episode_indices']
    assert len(ids) == 45
    images, truth, lengths = [], [], []
    with h5py.File(DATA, 'r') as f:
        for ep in ids:
            start, count = int(f['ep_offset'][ep]), int(f['ep_len'][ep])
            assert count >= 75 and np.isclose(float(f['dt'][ep]), .1)
            if 'action' in f:
                assert not np.any(f['action'][start:start + count])
            images.append(f['pixels'][start:start + 5])
            truth.append(f['state'][start:start + 75])
            lengths.append(float(f['length'][ep]))
    # Inference opens only the inputs file, containing no future pixels or labels.
    np.savez_compressed(OUT/'inputs.npz', pixels=np.stack(images), episode_ids=ids)
    np.savez_compressed(OUT/'truth.npz', state=np.stack(truth), length=lengths, episode_ids=ids)
    write('protocol.json', dict(dataset_sha256=sha(DATA), manifest_sha256=sha(MANIFEST),
          episodes=ids, dt=.1, anchor_raw_frame=4, horizon=70,
          inputs='five initial RGB frames, zero controls; no future RGB or state labels',
          training='none', readouts='all existing and frozen; no refit or test selection',
          old_lewm='all three undamped seed0/1/2 best checkpoints from the 35-epoch run',
          limitations=['LeWM trained on old fixed-length/partly driven data and different rendering.',
                      'LeHami trained on the target multi-length data for 100 epochs.',
                      'Old LeWM uses frozen ridge/MLP on context; LeHami uses its frozen official probes.',
                      'Transfer diagnostic, not a controlled model ranking.',
                      'Common frame4 anchor differs from original LeHami frame3.']))
    print('Prepared 45 aligned test episodes', flush=True)


def lewm():
    import torch
    sys.path.insert(0, str(OLD))
    from dpbench.models import load_checkpoint
    from dpbench.evaluate import Readout, decode_state
    def tensor_digest(state):
        digest = hashlib.sha256()
        for key, value in sorted(state.items()):
            value = value.detach().cpu().contiguous()
            digest.update(str((key, value.dtype, tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()
    def load_readouts(path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert set(saved) == {'ridge','mlp'}
        result = {}
        for key, value in saved.items():
            arrays = {k:value[k].numpy().copy() for k in ['mean','scale','ymean','yscale']}
            network, weights = None, None
            if key == 'ridge':
                weights = value['weights'].numpy().copy()
            else:
                network = torch.nn.Sequential(torch.nn.Linear(len(arrays['mean']),128),torch.nn.Tanh(),
                          torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,3))
                network.load_state_dict(value['network'], strict=True)
                network.eval().requires_grad_(False)
            result[key] = Readout(mode=key, network=network, weights=weights, **arrays)
        return result
    torch.set_num_threads(4)
    inputs = np.load(OUT/'inputs.npz')
    histories = inputs['pixels']
    root = OLD/'studies/independent_eval_20261029T072955Z/study/frozen/runs/server_47111'
    for seed in range(3):
        start = time.perf_counter()
        path = root / f'lewm_b0p0_seed{seed}'
        model_path, probe_path = path/'best.pt', path/'evaluation/readouts.pt'
        file_hashes = {str(p): sha(p) for p in [model_path, probe_path]}
        model, checkpoint = load_checkpoint(model_path, 'cuda')
        model.requires_grad_(False)
        probes = load_readouts(probe_path)
        before = tensor_digest(model.state_dict())
        predictions = {key: [] for key in probes}
        # Use original single-episode execution: batching failed the strict
        # decoded-trajectory equivalence check under this distribution shift.
        for offset in range(len(histories)):
            pixels = torch.from_numpy(histories[offset:offset+1].transpose(0,1,4,2,3).copy()).float().cuda()/255
            actions = torch.zeros(len(pixels), HORIZON, 1, device='cuda')
            features = model.rollout_features(pixels, actions).cpu().numpy()
            assert np.isfinite(features).all()
            for key, probe in probes.items():
                predictions[key].append(decode_state(probe.predict(features)))
        assert before == tensor_digest(model.state_dict())
        assert file_hashes == {str(p):sha(p) for p in [model_path, probe_path]}
        np.savez_compressed(OUT/f'lewm_seed{seed}.npz', **{k:np.concatenate(v) for k,v in predictions.items()})
        write(f'lewm_seed{seed}.json', dict(seed=seed, checkpoint_step=checkpoint['step'],
              training_data_sha256=checkpoint['dataset_sha256'], file_hashes=file_hashes,
              weights_unchanged=True, readouts_refitted=False, batch_size=1, seconds=time.perf_counter()-start,
              torch_version=str(torch.__version__), config=checkpoint['config']))
        print(f'Finished LeWM seed{seed}: {time.perf_counter()-start:.1f}s', flush=True)
        del model
        torch.cuda.empty_cache()


def lehami():
    import torch
    sys.path.insert(0, str(OFF/'LeHami/lehami/experiments/src'))
    import run_eval as re
    # Author evaluation was executed as a script, so its cached Probe class
    # was pickled under __main__ rather than run_eval.
    globals()['Probe'] = re.Probe
    from lehami.derivative import p_causal_last
    torch.set_num_threads(4)
    checkpoint = OFF/'data/checkpoints/lehami_sg_sep_sreg_joint_ms_conv_pendulum_multi_v2_sg_sep_conv_w1_multi_exp067_a0m/best.ckpt'
    probe_dir = OFF/'LeHami/lehami/experiments/results/exp067_a0m_lehami_w1_multi'
    names = {'knn_k1':'probe_k1_knn.pkl', 'mlp_gt':'probe_mlp.pkl', 'ridge_gt':'probe.pkl'}
    hashes = {str(p):sha(p) for p in [checkpoint]+[probe_dir/v for v in names.values()]}
    probes = {key:pickle.loads((probe_dir/name).read_bytes()) for key,name in names.items()}
    model, cfg = re.load_model(checkpoint, 'cuda')
    model.requires_grad_(False)
    before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    inputs = np.load(OUT/'inputs.npz')
    ratios = re._episode_dt_ratio(DATA, cfg)
    started = time.perf_counter()
    for anchor in [3,4]:
        predictions = {key:[] for key in probes}
        with torch.no_grad():
            for ep, history in zip(inputs['episode_ids'], inputs['pixels']):
                # Match the author's encoder batch shape without reading future
                # imagery. This W1 encoder is framewise in eval mode; padding
                # uses copies of an observed frame and its outputs are ignored.
                observed = history[:anchor+1]
                padded = np.concatenate([observed, np.repeat(observed[-1:],50-len(observed),axis=0)])
                pixels = re._preprocess_batch(padded, cfg.img_size, 'cuda')
                _, q = re._encode_episode(model, pixels)
                if ep == inputs['episode_ids'][0]:
                    alternate = padded.copy()
                    alternate[len(observed):] = observed[0]
                    _, qc = re._encode_episode(model, re._preprocess_batch(alternate,cfg.img_size,'cuda'))
                    assert torch.equal(q[:len(observed)],qc[:len(observed)]), 'padding affected observed latents'
                p0 = p_causal_last(q[anchor-3:anchor+1], float(cfg.model.fd_dt))
                h = None if ratios is None else torch.full((1,), float(ratios[ep]), device='cuda')
                if h is not None:
                    p0 = p0 / float(ratios[ep])
                qp = torch.cat([q[anchor], p0]).unsqueeze(0)
                action = re._zero_action_emb(model, cfg, 'cuda')
                features = [qp[0].cpu().numpy()]
                for _ in range(HORIZON):
                    qp = model.predictor(qp[:,None], action[:,None], h)[:,0]
                    features.append(qp[0].detach().cpu().numpy())
                features = np.stack(features)
                assert np.isfinite(features).all()
                for key, probe in probes.items():
                    x = features if probe.uses_p else features[:,:probe.latent_dim]
                    y = probe.ridge.predict(x)
                    predictions[key].append(np.stack([np.arctan2(y[:,0],y[:,1]),y[:,2]],-1))
        np.savez_compressed(OUT/f'lehami_anchor{anchor}.npz', **{k:np.stack(v) for k,v in predictions.items()})
        print(f'Finished LeHami anchor{anchor}', flush=True)
    assert all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items())
    assert hashes == {p:sha(p) for p in hashes}
    write('lehami.json', dict(seed=42, file_hashes=hashes, weights_unchanged=True,
          readouts_refitted=False, seconds=time.perf_counter()-started, torch_version=str(torch.__version__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','lewm','lehami'])
    globals()[parser.parse_args().mode]()
