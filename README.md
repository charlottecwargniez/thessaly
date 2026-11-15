# thessaly: Centaur experiments on JASMIN ORCHID

## Minitaur demo (task 3.2, step 1)

Loads [Minitaur-8B](https://huggingface.co/marcelbinz/Llama-3.1-Minitaur-8B), the small version of Centaur,
on one ORCHID GPU and runs the odd-one-out demo prompt from the Centaur repo's `run_minimal.py`.

| File | What it does |
| --- | --- |
| `setup_jasmin.sh` | One-off: creates the conda env and downloads the model into the group workspace. Run on a sci server. |
| `minitaur_demo.py` | Loads the model in BF16 and prints the sampled choice, the H/Y/E probabilities, load time and GPU memory. |
| `minitaur_demo.sbatch` | Slurm job: 1 GPU on the `orchid` partition, 1 hour, offline mode. |

### Run it (when you're ready)

```bash
ssh <jasmin-sci-server>                     # e.g. sci-ph-02, via the JASMIN login node
bash setup_jasmin.sh                        # once; ~16 GB download into /gws/ssde/j25b/iecdt/cwarg/hf_cache
cp minitaur_demo.py minitaur_demo.sbatch /gws/ssde/j25b/iecdt/cwarg/thessaly/
cd /gws/ssde/j25b/iecdt/cwarg/thessaly && sbatch minitaur_demo.sbatch
cat minitaur_demo_<jobid>.out
```

### What to expect

- Model weights take about 16 GB of GPU memory, so it fits easily on one A100 40GB.
- The model should put most of its probability on H, Y or E. The sum line shows how much.
- The probabilities are the interesting part: they are Minitaur's prediction of how a human would split on this triplet,
  not a "correct" answer.

### Notes

- Env and cache live in the group workspace because home is at 79G of 100G.
- Never run the model on the login or sci servers; only inside the Slurm job.
