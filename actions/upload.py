import glob
import os
import sys

script = os.path.abspath(sys.argv[0])

# go up one directories to get the source directory
# (this script is in Sire/actions/)
srcdir = os.path.dirname(os.path.dirname(script))

print(f"sire source is in {srcdir}\n")

# Get the anaconda token to authorise uploads
if "ANACONDA_TOKEN" in os.environ:
    conda_token = os.environ["ANACONDA_TOKEN"]
else:
    conda_token = "TEST"

# get the root conda directory
conda = os.environ["CONDA"]

# Set the path to the conda-bld directory.
conda_bld = os.path.join(conda, "envs", "build_env", "conda-bld")

print(f"conda_bld = {conda_bld}")

# Find the packages to upload
pkg_lycean = glob.glob(os.path.join(conda_bld, "*-*", "lyceanem-*.tar.bz2"))

if len(pkg_lycean) == 0:
    print("No lyceanem packages to upload?")
    sys.exit(-1)

packages = pkg_lycean


print(f"Uploading packages:")


packages = " ".join(packages)


def run_cmd(cmd):
    import subprocess

    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    return str(p.stdout.read().decode("utf-8")).lstrip().rstrip()


gitdir = os.path.join(srcdir, ".git")

tag = run_cmd(f"git --git-dir={gitdir} --work-tree={srcdir} tag --contains")
branch = run_cmd(f"git branch --show-current")
# if the branch is master, then this is a main release
#if tag is not None and tag.lstrip().rstrip() != "":
#    print(f"\nTag {tag} is set. This is a 'main' release.")
#    label = "--label main"
#else:
#    # this is a development release
#    print("\nNo tag is set. This is a 'devel' release.")
#    label = "--label dev"
if branch=="master":
    print(f"\nBranch {branch} is set. This is a 'main' release.")
    label = "--label main"
else:
    # this is a development release
    print(f"\nBranch {branch} is set. This is a 'devel' release.")
    label = "--label dev"

# Upload the packages to the michellab channel on Anaconda Cloud.
cmd = (
    f"anaconda --token {conda_token} upload --user LyceanEM {label} --force {packages}"
)

print(f"\nUpload command:\n\n{cmd}\n")

# Label release packages with main and dev so that dev is at least as new as
# main. Only need to uncomment the libcpuid and kcombu package uploads when
# there new versions are released.
if conda_token == "TEST":
    print("Not uploading as the ANACONDA_TOKEN is not set!")
    sys.exit(-1)

output = run_cmd(cmd)

print(output)

print("Package uploaded!")
