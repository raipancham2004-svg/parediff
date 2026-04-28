# Kaggle Training Setup — Step-by-Step

You'll do this **once** to bootstrap, then it's just "open notebook → Run All → close laptop → come back in 12 hr."

## Step 1: Create the 3 free accounts (~10 min total)

### Kaggle (training runs here)
1. Go to https://www.kaggle.com/account/login
2. Sign in with your Google account
3. **Phone-verify your account** (required to unlock GPU access — Settings → Account → Phone Verification)
4. Skip if you already have one

### Weights & Biases (training logs)
1. Go to https://wandb.ai/site
2. Sign up with GitHub (one click)
3. Go to Settings → API Keys → copy your API key
4. **Save the key somewhere safe** — we'll paste it into Kaggle

### Hugging Face (checkpoint storage — free unlimited)
1. Go to https://huggingface.co/join
2. Sign up with GitHub
3. Settings → Access Tokens → New token (write permission) → copy
4. **Save the token somewhere safe**

---

## Step 2: Create the GitHub repo (5 min)

You probably have GitHub already. Create a new repo:

```bash
# Tell me your GitHub username, then I'll give exact commands.
# Repo name suggestion: parediff
# Visibility: Public (so Kaggle can clone it without auth)
```

Push our code to it (I'll write the exact `git` commands once you share your username).

---

## Step 3: Add Kaggle Secrets (so the notebook can authenticate)

In your Kaggle notebook (we'll create it in Step 4):
1. Add-ons → Secrets → Add a new secret
2. Add two secrets:
   - Label: `WANDB_KEY`, value: your wandb API key from Step 1
   - Label: `HF_TOKEN`, value: your Hugging Face token from Step 1
3. Toggle both to "Attach" for the current notebook

---

## Step 4: Create the Kaggle notebook

1. Go to https://www.kaggle.com/code → New Notebook
2. Click the ⋮ menu → Settings:
   - **Accelerator: GPU P100** (16 GB) ← important
   - **Internet: ON** ← important (so it can pip install + git clone)
   - **Persistence: Variables and files** ← saves output between sessions
3. **Copy-paste the contents of `train_on_kaggle.ipynb`** (I'll provide this)
4. Click "Save Version" → "Quick Save" to save

---

## Step 5: Run training

1. Open your notebook → click **"Run All"** (top toolbar)
2. The first run will take ~30 min to install deps + download data
3. Then training kicks off — you can close the browser
4. Come back in ~12 hr when session expires
5. Notebook output (including checkpoints) is auto-saved to `/kaggle/working/`
6. Restart session → notebook auto-resumes from last checkpoint

---

## Step 6: Monitor live (optional)

While training runs, the wandb dashboard shows real-time loss curves:
- Open https://wandb.ai/<your-username>/parediff
- Loss should go DOWN → if it's flat or NaN, ping me
- Share the wandb URL with me in chat → **I'll watch progress and tell you when to stop**

---

## Total time commitment

| Phase | Your effort | Wall clock |
|---|---|---|
| Account setup | 10 min | 10 min |
| GitHub push (with my help) | 5 min | 5 min |
| Kaggle notebook setup | 5 min | 5 min |
| Training | ~3 sessions × 5 min clicking | ~36 hr (mostly your laptop is OFF) |
| **Total active work** | **~30 min** | **~2 days wall clock** |

You will spend more time WAITING than DOING. Perfect for a busy intern + final-year student.
