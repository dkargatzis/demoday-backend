import httpx
import re
from typing import Dict, Any

async def extract_github_context(repo_url: str) -> Dict[str, Any]:
    match = re.search(r'github\.com/([^/]+)/([^/]+)', repo_url)
    if not match:
        return {'raw_repo': repo_url, 'readme': '', 'package_json': {}}

    owner, repo = match.group(1), match.group(2).replace('.git', '')
    headers = {'User-Agent': 'DemoDayAgent/1.0'}

    readme_content = ''
    package_json = {}

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        for branch in ['main', 'master']:
            try:
                res = await client.get(
                    f'https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md',
                    headers=headers
                )
                if res.status_code == 200:
                    readme_content = res.text[:8000]
                    break
            except Exception:
                pass

        for branch in ['main', 'master']:
            try:
                pkg_res = await client.get(
                    f'https://raw.githubusercontent.com/{owner}/{repo}/{branch}/package.json',
                    headers=headers
                )
                if pkg_res.status_code == 200:
                    package_json = pkg_res.json()
                    break
            except Exception:
                pass

    return {
        'owner': owner,
        'repo': repo,
        'readme': readme_content,
        'name': package_json.get('name', repo),
        'description': package_json.get('description', ''),
        'dependencies': list(package_json.get('dependencies', {}).keys())[:15]
    }

async def probe_app_url(app_url: str) -> Dict[str, Any]:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) DemoDayProbe/1.0'}
    meta_info = {'title': '', 'description': '', 'status': 0}
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True, verify=False) as client:
            res = await client.get(app_url, headers=headers)
            meta_info['status'] = res.status_code
            text = res.text
            title_m = re.search(r'<title>(.*?)</title>', text, re.IGNORECASE)
            if title_m:
                meta_info['title'] = title_m.group(1).strip()
            desc_m = re.search(r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*name=["\']description["\']', text, re.IGNORECASE)
            if not desc_m:
                desc_m = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']', text, re.IGNORECASE)
            if desc_m:
                meta_info['description'] = desc_m.group(1).strip()
    except Exception as e:
        meta_info['error'] = str(e)
    return meta_info
