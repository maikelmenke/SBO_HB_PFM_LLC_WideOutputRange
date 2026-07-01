"""
Code Ocean entry point — set this file as "File to Run".

Toggle each example below (True = run, False = skip).
Source code and dependencies are pre-installed by environment/postInstall.
"""

import os
import subprocess
import sys
import urllib.request
import zipfile

# ── Configure which examples to run ──────────────────────────────────────────
RUN_01_BASIC     = True    # Basic single transient simulation
RUN_02_SS        = True    # Simulate until steady state (block-by-block)
RUN_03_PEAK      = True    # Peak-gain frequency mapping
RUN_04_VO_SINGLE = True    # Single operating point — convergence plot
RUN_05_VO_BATCH  = True    # Batch operating-point evaluation
RUN_GA           = False   # GA optimisation — long-running, off by default
RUN_PLOT_GA      = False   # Plot GA results (reads CSVs from /results/ga/)
# ─────────────────────────────────────────────────────────────────────────────

os.environ.setdefault('MPLBACKEND', 'Agg')

# Prefer /opt/llc (pre-installed by postInstall); fall back to /tmp/llc if the
# environment was not rebuilt yet after postInstall was updated.
if os.path.isdir('/opt/llc/llc_sim'):
    CODE = '/opt/llc'
else:
    CODE = '/tmp/llc'
    if not os.path.isdir(os.path.join(CODE, 'llc_sim')):
        print("postInstall has not run yet — downloading source from GitHub...")
        zip_url = ('https://github.com/maikelmenke/'
                   'SBO_HB_PFM_LLC_WideOutputRange/archive/refs/heads/codeocean.zip')
        zip_path = '/tmp/llc_src.zip'
        req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(zip_path, 'wb') as out:
            out.write(resp.read())
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall('/tmp/')
        os.rename('/tmp/SBO_HB_PFM_LLC_WideOutputRange-codeocean', CODE)
        print("Installing Python dependencies...")
        _pip = [sys.executable, '-m', 'pip', 'install', '-q', '--break-system-packages']
        subprocess.run(_pip + ['-r', os.path.join(CODE, 'requirements.txt')], check=True)
        subprocess.run(_pip + [CODE], check=True)
        subprocess.run(['pyspice-post-installation', '--install-ngspice-dll'], check=True)

    # Install libngspice0 v36 if not already present
    _so0 = '/usr/lib/x86_64-linux-gnu/libngspice.so.0'
    if not os.path.exists(_so0):
        print("Installing libngspice0 v36...")
        subprocess.run(['apt-get', 'update', '-qq'], check=True)
        deb_url = ('http://archive.ubuntu.com/ubuntu/pool/universe/n/ngspice/'
                   'libngspice0_36+ds-1_amd64.deb')
        deb_path = '/tmp/libngspice0_v36.deb'
        req_deb = urllib.request.Request(deb_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_deb) as resp, open(deb_path, 'wb') as out:
            out.write(resp.read())
        subprocess.run(['apt-get', 'install', '-y', '--allow-downgrades', deb_path], check=True)
    _so = '/usr/lib/x86_64-linux-gnu/libngspice.so'
    if os.path.exists(_so0) and not os.path.exists(_so):
        os.symlink(_so0, _so)

print(f"Using source from: {CODE}")

# LD_LIBRARY_PATH so dlopen finds libngspice.so (v36) in every subprocess
_LIB_DIR = '/usr/lib/x86_64-linux-gnu'
os.environ['LD_LIBRARY_PATH'] = (
    _LIB_DIR + ':' + os.environ.get('LD_LIBRARY_PATH', '')
).rstrip(':')

os.chdir(CODE)

# Redirect outputs/ → /results/ so all scripts save there automatically
try:
    os.symlink('/results/', os.path.join(CODE, 'outputs'))
except FileExistsError:
    pass


def _run(label, *cmd):
    print(f'\n{"=" * 60}')
    print(f'  {label}')
    print('=' * 60)
    subprocess.run(cmd, cwd=CODE, check=True)


if RUN_01_BASIC:
    _run('Example 1 — Basic single transient simulation',
         sys.executable, 'Example.py')

if RUN_02_SS:
    _run('Example 2 — Simulate until steady state',
         sys.executable, 'Example_Simulate_Until_SS.py')

if RUN_03_PEAK:
    _run('Example 3 — Peak-gain frequency mapping',
         sys.executable, 'Example_Find_Vpeak_ss.py')

if RUN_04_VO_SINGLE:
    _run('Example 4 — Single operating point, convergence plot',
         sys.executable, 'Example_find_Vo_target_single.py')

if RUN_05_VO_BATCH:
    _run('Example 5 — Batch operating-point evaluation',
         sys.executable, 'Example_Find_Vo_target.py')

if RUN_GA:
    os.makedirs('/results/ga', exist_ok=True)
    try:
        os.symlink('/results/ga', os.path.join(CODE, 'opt/outputs'))
    except FileExistsError:
        pass
    _run('GA Optimisation  (see /results/ga/ for log and CSV)',
         sys.executable, os.path.join(CODE, 'opt/main_ga.py'))

if RUN_PLOT_GA:
    os.makedirs('/results/ga', exist_ok=True)
    try:
        os.symlink('/results/ga', os.path.join(CODE, 'opt/outputs'))
    except FileExistsError:
        pass
    _run('GA Plot  (reads CSVs from /results/ga/, saves PDFs there)',
         sys.executable, os.path.join(CODE, 'opt/plot_ga.py'))

print('\nDone. All results saved to /results/')
