import os
from dotenv import load_dotenv
from github import Github
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ================= CONFIG =================
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

OUTPUT_DIR = Path("data")
GITHUB_FILE = OUTPUT_DIR / "github_data.txt"

if not GITHUB_TOKEN:
    raise ValueError("❌ Missing GITHUB_TOKEN in .env")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= STEP 1: Fetch GitHub Repositories =================
def fetch_repos():
    g = Github(GITHUB_TOKEN)
    user = g.get_user()
    repo_results = []

    def process_repo(repo):
        try:
            info = [
                f"===== REPO: {repo.name} =====\n",
                f"URL: {repo.html_url}\n"
            ]
            try:
                readme = repo.get_readme()
                info.append("\n--- README ---\n")
                info.append(readme.decoded_content.decode("utf-8"))
                info.append("\n--- END README ---\n")
            except Exception:
                info.append("(No README)\n")
            info.append("===== END REPO =====\n\n")
            return "".join(info)
        except Exception as e:
            return f"⚠️ Failed repo {repo.name}: {e}\n"

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_repo, repo) for repo in user.get_repos()]
        for f in as_completed(futures):
            repo_results.append(f.result())

    return repo_results


# ================= STEP 2: Save to File =================
def save_repos(repo_texts, output_file=GITHUB_FILE):
    with open(output_file, "w", encoding="utf-8") as f:
        for repo in repo_texts:
            f.write(repo)
    print(f"✅ GitHub data saved → {output_file}")


# ================= MAIN =================
def run_github_fetch():
    print("🔄 Fetching GitHub repositories...")
    repos = fetch_repos()
    save_repos(repos)


if __name__ == "__main__":
    run_github_fetch()
