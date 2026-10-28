"""Track the same motor unit across contractions, using MUedit's own method.

Ported from `simonavrillon/MUdictionnary` (Avrillon et al. 2024, `MUdict.m`) and
used by `notebooks/hysteresis_repeatability.ipynb` to follow each decomposed
motor unit through the 10-80 % MVC contractions, so its hysteresis can be
recomputed contraction by contraction rather than only as an average.

Tracking is filter reconstruction, re-application and spike coincidence -- not
MUAP-waveform correlation, which aliases units:

1. `level_step` reconstructs each unit's separation filter from the raw EMG and
   its discharge times (extend, then spike-triggered average), and re-applies
   the filters carried over from earlier contractions.
2. `match` pairs a re-applied filter with a natively decomposed unit when their
   discharge times coincide (rate of agreement >= `thr`).
3. `process_subject` walks the contractions in order, growing a dictionary of
   units per subject.

Every setting the notebook configures is a parameter here, so a call made twice
with the same arguments returns the same tracks.
"""

import re
import glob
import os
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter1d


# --------------------------------------------------------------------------
# Reading the recordings
# --------------------------------------------------------------------------

def resolve(subj, lvl, raw_dir):
    """Path to one subject's recording at one contraction level.

    Prefers a manually edited decomposition when one exists.
    """
    c = glob.glob(f"{raw_dir}/S{subj}_{lvl}_DF*.mat")
    if not c:
        return None
    n = [x for x in c if "edited_new" in x]
    return n[0] if n else sorted(c)[0]


def native_units(m, g):
    """Discharge times of the units decomposed natively on grid `g`.

    Units with fewer than 5 discharges are dropped. MATLAB indices are
    converted to 0-based and clipped to the length of the recording.
    """
    dt = np.atleast_2d(m['edition'].Dischargetimes)
    n = np.asarray(m['signal'].data).shape[1]
    out = []
    if g >= dt.shape[0]:
        return out
    for j in range(dt.shape[1]):
        st = np.atleast_1d(np.asarray(dt[g, j])).ravel()
        st = st[np.isfinite(st)].astype(int) - 1
        st = st[(st >= 0) & (st < n)]
        if st.size >= 5:
            out.append(st)
    return out


# --------------------------------------------------------------------------
# The MUedit steps
# --------------------------------------------------------------------------

def extend(eY, ex):
    """Delay-embed the EMG by `ex` samples, as MUedit's extension step does."""
    r, cN = eY.shape
    out = np.zeros((r * ex, cN + ex - 1), np.float32)
    for m in range(ex):
        out[m * r:(m + 1) * r, m:cN + m] = eY
    return out


def level_step(emg, nat, Dfil, nbext, fs):
    """Reconstruct this contraction's filters, and re-apply the carried-over ones.

    `getMUfilters`: each native unit's filter is the spike-triggered average of
    the extended EMG. `getPulseT`: filters from earlier contractions are
    re-applied as ``s = (STA^T . C^-1) . eSIG``, squared-with-sign, peak-picked,
    and split into spikes and noise by a two-centre k-means on the peak heights.

    Returns (filters for this contraction, re-applied spike trains).
    """
    ex = round(nbext / emg.shape[0])
    eS = extend(emg, ex)
    T = emg.shape[1]
    Fn = np.zeros((eS.shape[0], len(nat)), np.float32)
    for i, st in enumerate(nat):
        Fn[:, i] = eS[:, st].sum(1)                         # getMUfilters (STA)
    reapp = []
    if Dfil is not None and Dfil.shape[1] > 0:
        C = (eS @ eS.T) / eS.shape[1]
        iC = np.linalg.pinv(C).astype(np.float32)
        S = (Dfil.T @ iC) @ eS                              # getPulseT: sources
        for k in range(S.shape[0]):
            pt = S[k, :T]
            pt = pt * np.abs(pt)
            pk, _ = find_peaks(pt, distance=round(fs * 0.02))
            if len(pk) < 3:
                reapp.append(np.array([], int))
                continue
            vals = pt[pk]
            c = np.array([vals.min(), vals.max()])
            for _ in range(12):
                lab = (np.abs(vals[:, None] - c[None, :])).argmin(1)
                for tt in (0, 1):
                    if (lab == tt).any():
                        c[tt] = vals[lab == tt].mean()
            reapp.append(pk[lab == int(np.argmax(c))])
        del C, iC, S
    del eS
    return Fn, reapp


def match(reapp, nat, T, thr, jit):
    """Pair re-applied filters with natively decomposed units by spike coincidence.

    Two trains agree when their discharges fall within `jit` samples of each
    other; the rate of agreement is the count of shared discharges over the
    longer train. Pairs are taken best-first, each unit used at most once, and
    anything below `thr` is left unmatched.
    """
    nD, nN = len(reapp), len(nat)
    if nD == 0 or nN == 0:
        return []
    nb = np.zeros((nN, T), np.float32)
    for j, sp in enumerate(nat):
        s = sp[(sp >= 0) & (sp < T)]
        nb[j, s] = 1
    nd = maximum_filter1d(nb, size=2 * jit + 1, axis=1)
    rb = np.zeros((nD, T), np.float32)
    for i, sp in enumerate(reapp):
        s = sp[(sp >= 0) & (sp < T)]
        if len(s):
            rb[i, s] = 1
    common = rb @ nd.T
    lenR = np.array([len(s) for s in reapp])[:, None]
    lenN = np.array([len(s) for s in nat])[None, :]
    roa = common / np.maximum(np.maximum(lenR, lenN), 1)
    pairs = []
    ud = set()
    un = set()
    for flat in np.argsort(-roa, axis=None):
        i, j = divmod(int(flat), nN)
        if roa[i, j] < thr:
            break
        if i in ud or j in un:
            continue
        ud.add(i); un.add(j); pairs.append((i, j))
    return pairs


def process_subject(subj, raw_dir, levels, nbext, fs, thr, jit):
    """Track one subject's motor units across every contraction level available.

    Walks the levels in order, carrying each unit's filter forward and matching
    it against the next contraction's native decomposition. Returns one record
    per tracked unit with, for each contraction it appears in, the recruitment
    and de-recruitment force and the resulting hysteresis (DRT/RT - 1).
    """
    files = {lvl: resolve(subj, lvl, raw_dir) for lvl in levels
             if resolve(subj, lvl, raw_dir)}
    mats = {lvl: sio.loadmat(f, squeeze_me=True, struct_as_record=False)
            for lvl, f in files.items()}
    present = sorted(mats, key=int)
    data = {lvl: np.asarray(mats[lvl]['signal'].data, np.float32) for lvl in present}
    force = {lvl: np.asarray(mats[lvl]['signal'].path, float) for lvl in present}
    ngrid = min(data[lvl].shape[0] // 64 for lvl in present)
    chains = []
    for g in range(ngrid):
        natc = {lvl: native_units(mats[lvl], g) for lvl in present}
        D = []
        for lvl in present:
            nat = natc[lvl]
            if not nat:
                continue
            emg = data[lvl][g * 64:(g + 1) * 64, :]
            T = emg.shape[1]
            Dfil = np.stack([u['filter'] for u in D], axis=1) if D else None
            Fn, reapp = level_step(emg, nat, Dfil, nbext, fs)
            matched = set()
            if D:
                for di, ni in match(reapp, nat, T, thr, jit):
                    D[di]['levels'][lvl] = ni
                    D[di]['filter'] = Fn[:, ni]
                    matched.add(ni)
            for ni in range(len(nat)):
                if ni not in matched:
                    D.append(dict(filter=Fn[:, ni], levels={lvl: ni}))
        for u in D:
            rec = {}
            for lvl, ni in u['levels'].items():
                sp = natc[lvl][ni]
                f = force[lvl]
                rt = f[sp[0]]
                drt = f[sp[-1]]
                rec[int(lvl)] = dict(RT=float(rt), DRT=float(drt),
                                     hyst=float(drt / rt - 1) if abs(rt) > 1e-2 else np.nan,
                                     n=int(sp.size))
            chains.append(dict(subject=subj, grid=g, levels=rec))
    return chains


# --------------------------------------------------------------------------
# The reported values, for validation
# --------------------------------------------------------------------------

def read_xlsx_hyst(path):
    """Hysteresis values as reported in `Data - Hysteresis .xlsx`, keyed by subject.

    Reads the sheet directly from the zip container so that no spreadsheet
    library is needed. One column per subject, rows 3-221.
    """
    z = zipfile.ZipFile(path)
    NS = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    sst = []
    t = ET.fromstring(z.read('xl/sharedStrings.xml'))
    for si in t.findall('a:si', NS):
        sst.append("".join(n.text or "" for n in si.iter('{%s}t' % NS['a'])))
    sh = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    G = {}
    for row in sh.iter('{%s}row' % NS['a']):
        for c in row.findall('a:c', NS):
            ref = c.get('r')
            col = re.match(r'[A-Z]+', ref).group()
            r = int(re.search(r'\d+', ref).group())
            v = c.find('a:v', NS)
            if v is not None:
                G[(r, col)] = sst[int(v.text)] if c.get('t') == 's' else v.text
    hcol = {1: 'C', 2: 'F', 3: 'I', 4: 'L', 5: 'O', 6: 'R', 7: 'U', 8: 'X'}
    out = {}
    for s, col in hcol.items():
        vals = []
        for r in range(3, 222):
            try:
                vals.append(float(G.get((r, col))))
            except (TypeError, ValueError):
                pass
        out[s] = np.array(vals)
    return out
