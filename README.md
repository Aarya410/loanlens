# LoanLens: ML loan-approval predictor with a fully automated CI/CD pipeline

A web app where a Gradient Boosting model predicts the probability that a loan application is approved,
and explains which inputs helped or hurt. Every `git push` to `main` is tested, containerised and deployed
automatically, with no manual step.

## Architecture

```
git push -> GitHub Actions / GitLab CI
              |-- 1. TEST   ruff lint, train model, pytest
              |-- 2. BUILD  Docker image (model trained inside), pushed to GHCR / GitLab Registry
              '-- 3. DEPLOY Render deploy hook for the exact commit -> smoke test checks /api/health
                            reports the same commit SHA before the pipeline turns green
```

| Layer | Tech |
|---|---|
| ML | scikit-learn GradientBoostingClassifier, pandas, joblib |
| API | FastAPI + Pydantic validation |
| UI | Single-page HTML/CSS/JS served by FastAPI (light and dark mode) |
| Container | Docker |
| CI/CD | GitHub Actions (`.github/workflows/ci-cd.yml`) and GitLab CI (`.gitlab-ci.yml`) |
| Hosting | Render (`render.yaml`), free tier |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python train.py          # creates models/model.joblib
pytest -q                # run tests
uvicorn app.main:app --reload
```
Open http://localhost:8000. API docs are at `/docs`.

With Docker: `docker build -t loanlens . && docker run -p 8000:8000 loanlens`

## One-time deployment setup (about 10 minutes)

1. **Push to GitHub**
   ```bash
   git init && git add . && git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<you>/loanlens.git
   git push -u origin main
   ```
   The pipeline starts immediately. The `test` and `build` jobs work with no extra setup.
2. **Create the host on Render**: New > Blueprint > pick this repo (it reads `render.yaml`).
   Open the service > Settings > copy the **Deploy Hook URL**.
3. **Add two GitHub secrets** (Repo > Settings > Secrets and variables > Actions):
   - `RENDER_DEPLOY_HOOK`: the deploy hook URL
   - `APP_URL`: your app URL, e.g. `https://loanlens.onrender.com` (no trailing slash)
4. Make any commit and push. Watch the **Actions** tab: test, build, deploy, then smoke test.
   The footer of the live site shows the commit SHA, which changes after each push.

Using GitLab instead? Push to GitLab and add `RENDER_DEPLOY_HOOK` and `APP_URL` under Settings > CI/CD > Variables.
`.gitlab-ci.yml` does the same three stages.

## Demo script for your viva

1. Change the page title in `app/static/index.html`, commit, push.
2. Show the pipeline running in the Actions tab.
3. Refresh the live site: the new title appears and the footer shows the new commit SHA.
4. Break a test on purpose and push: the pipeline fails and the broken build never deploys.

## About the data

The model trains on 8,000 synthetic applicant records generated in `app/model.py` using realistic
rules (credit score, debt burden, income, age plus tenure) with noise. For a stronger submission, replace
`generate_data()` with a real dataset (e.g. the Kaggle "Loan Prediction" dataset) and keep the rest unchanged.
Be upfront about this in your report, and the reported accuracy is therefore for the synthetic data.

## API

- `GET /api/health` returns status, deployed commit, build time and model metrics.
- `POST /api/predict` accepts the 8 applicant fields and returns probability, verdict, EMI estimate and top factors.

## Ideas to extend it

Add model versioning with MLflow, a scheduled retraining workflow, Prometheus metrics, or a staging environment deployed from pull requests.
