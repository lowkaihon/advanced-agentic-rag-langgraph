## Python `__pycache__` Best Practices

The `__pycache__` directory is Python's bytecode cache mechanism that stores compiled `.pyc` files to speed up module imports. While beneficial for performance, improper management can lead to stale bytecode issues during development. Here's a comprehensive guide to best practices.

### Should You Add `__pycache__` to `.gitignore`?

**Yes, always add `__pycache__` to your `.gitignore`**. This is a universal best practice in the Python community for several reasons:[^1_1][^1_2][^1_3]

- Bytecode files are **version-specific** and architecture-dependent[^1_4][^1_5]
- They can be regenerated automatically from source code
- Including them in version control bloats your repository unnecessarily
- Different team members may use different Python versions, causing conflicts[^1_4]

Add this to your `.gitignore`:

```
__pycache__/
*.pyc
*.pyo
```


### Preventing Stale Bytecode Cache Issues

Stale bytecode occurs when Python uses cached `.pyc` files that don't reflect recent source code changes. This typically happens in these scenarios:[^1_6][^1_7]

**When to delete `__pycache__`:**

- **After renaming or moving Python modules** - Python 3.2+ largely fixed this for cached bytecode, but standalone `.pyc` files can still cause issues[^1_7]
- **When debugging import problems** - If modules aren't updating as expected[^1_8][^1_9]
- **After switching git branches** with structural changes[^1_9]
- **When you suspect cached bytecode is causing bugs**[^1_10][^1_11]

**Quick cleanup commands:**

```bash
# Remove all __pycache__ directories recursively
find . -type d -name "__pycache__" -exec rm -r {} +

# Remove all .pyc files
find . -type f -name "*.pyc" -delete
```


### Cache Invalidation Mechanisms

Python uses two methods for determining when to regenerate bytecode:[^1_12][^1_13][^1_14]

**1. Timestamp-based invalidation (default)**

- Compares modification time of `.py` file with cached `.pyc`[^1_15]
- Fast but can miss updates when filesystem timestamps are coarse[^1_14][^1_6]
- Problematic for reproducible builds[^1_13][^1_16]

**2. Hash-based invalidation (Python 3.7+)**[^1_13][^1_14]

- Uses SHA-256 hash of source file instead of timestamp
- More reliable but slightly slower (still much faster than recompiling)[^1_16]
- Ideal for content-based build systems and Docker images[^1_17][^1_18]
- Enables deterministic builds[^1_13]

Generate hash-based `.pyc` files:

```bash
python -m compileall --invalidation-mode checked-hash
```


### Recommended Development Approaches

**Option 1: Disable bytecode generation during development**[^1_19][^1_8]

This is commonly used in Docker containers, CI/CD pipelines, and when actively developing:

```bash
# Environment variable
export PYTHONDONTWRITEBYTECODE=1

# Or command-line flag
python -B your_script.py
```

**Benefits:**

- No stale bytecode issues
- Cleaner workspace[^1_20]
- Reduced Docker image size in some contexts[^1_8]

**Drawbacks:**

- Slightly slower import times (usually negligible for development)[^1_8]
- No performance benefit on repeated runs

**Option 2: Use centralized cache directory (Python 3.8+)**[^1_21][^1_22][^1_12]

The `PYTHONPYCACHEPREFIX` environment variable stores all bytecode in a single location outside your project:

```bash
export PYTHONPYCACHEPREFIX="$HOME/.cache/cpython/"
```

**Benefits:**

- Clean project directories[^1_23]
- Easy to clear all cache at once[^1_12]
- No `__pycache__` folders scattered through your codebase[^1_24][^1_21]
- Can use separate disk for parallel I/O performance[^1_12]

**Drawbacks:**

- Requires Python 3.8+[^1_25][^1_22]
- Must set environment variable consistently[^1_25]

Add to your `~/.bashrc` or `~/.zshrc` to make it permanent.[^1_21][^1_23]

**Option 3: Keep default behavior with good `.gitignore`**[^1_11]

For most projects, the default `__pycache__` behavior works well:

- Performance benefits during development[^1_3][^1_11]
- Python handles invalidation automatically[^1_26][^1_15]
- Just ensure proper `.gitignore` configuration[^1_2][^1_1]


### Advanced Considerations

**Pytest and testing:**
The `pytest-remove-stale-bytecode` plugin automatically removes stale bytecode before test runs, preventing deleted modules from being accidentally imported.[^1_27]

**Pre-commit hooks:**
Automate bytecode cleanup by adding to `.pre-commit-config.yaml` or creating custom git hooks, though this is less common than simply using `.gitignore`.[^1_28]

**Production deployments:**

- **Keep bytecode** in production for faster startup times[^1_29][^1_11]
- Use hash-based `.pyc` for reproducible builds[^1_17][^1_13]
- SELinux and similar security policies may prevent runtime bytecode generation, making pre-compiled bytecode necessary[^1_30]

**Module reloading in REPL:**
When actively developing and testing in a Python REPL, use `importlib.reload()` or restart the REPL entirely, as imports are cached in memory regardless of `.pyc` files.[^1_31]

### Version Compatibility

`.pyc` files contain a **magic number** that identifies the Python version. Files are **not compatible** across major or even minor Python versions. Python automatically:[^1_5][^1_4]

- Ignores incompatible `.pyc` files[^1_4]
- Regenerates bytecode if the source is available and permissions allow[^1_30]
- Stores version information in the filename (e.g., `module.cpython-39.pyc`)[^1_15][^1_5]

This means mixed-version environments need isolation via virtual environments to prevent conflicts.[^1_4]

### Summary of Best Practices

1. **Always add `__pycache__/` to `.gitignore`**[^1_1][^1_2][^1_3]
2. **Use `PYTHONDONTWRITEBYTECODE=1`** in Docker containers and CI/CD[^1_32][^1_33]
3. **Use `PYTHONPYCACHEPREFIX`** (Python 3.8+) for cleaner local development[^1_22][^1_23][^1_21]
4. **Delete `__pycache__` when debugging import issues** or after major refactoring[^1_9][^1_7][^1_8]
5. **Keep bytecode enabled in production** for performance[^1_11][^1_29]
6. **Use hash-based `.pyc`** for reproducible builds and deployments[^1_17][^1_13]
7. **Restart your Python REPL** rather than relying on `importlib.reload()` during active development[^1_31]
<span style="display:none">[^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71]</span>

<div align="center">⁂</div>

[^1_1]: https://towardsdatascience.com/pycache-python-991424aabad8/

[^1_2]: https://www.tracedynamics.com/git-ignore-pycache/

[^1_3]: https://www.pythonsnacks.com/p/python-pycache-directory

[^1_4]: https://stackoverflow.com/questions/2263356/are-python-2-5-pyc-files-compatible-with-python-2-6-pyc-files

[^1_5]: https://peps.pythondiscord.com/pep-3147/

[^1_6]: https://bugs.python.org/issue31772

[^1_7]: https://python-notes.curiousefficiency.org/en/latest/python_concepts/import_traps.html

[^1_8]: https://docs.kanaries.net/topics/Python/pycache

[^1_9]: https://www.cmarix.com/qanda/python-module-not-updating-after-import/

[^1_10]: https://python.plainenglish.io/what-is-pycache-️-22c29439f87f

[^1_11]: https://www.ghacks.net/2023/09/07/what-is-__pycache__-and-can-you-delete-it/

[^1_12]: https://realpython.com/python-pycache/

[^1_13]: https://peps.python.org/pep-0552/

[^1_14]: https://docs.python.org/3/whatsnew/3.7.html

[^1_15]: https://raul.dev/post/python_optimized_files/

[^1_16]: https://news.ycombinator.com/item?id=23367559

[^1_17]: https://lists.fedoraproject.org/archives/list/python-devel@lists.fedoraproject.org/thread/Y2QRYFOWXUZWEUOHX7JKPFO2A3N23ATK/

[^1_18]: https://harveyli.me/whats-new-in-python-3-7/

[^1_19]: https://stackoverflow.com/questions/154443/how-to-avoid-pyc-files

[^1_20]: https://python-docs.readthedocs.io/en/latest/writing/gotchas.html

[^1_21]: https://stackoverflow.com/questions/28991015/remove-pycache-folders-and-pyc-files-from-python-project

[^1_22]: https://pradyunsg-cpython-lutra-testing.readthedocs.io/en/latest/whatsnew/3.8.html

[^1_23]: https://tech.serhatteker.com/post/2022-06/remove_python_pycache_pyc/

[^1_24]: https://stackoverflow.com/questions/78725383/is-it-possible-to-make-pycache-generate-in-a-single-centralized-folder

[^1_25]: https://stackoverflow.com/questions/58953969/python-get-the-python3-8-new-feature-pythonpycacheprefix-working-for-web-appli

[^1_26]: https://stackoverflow.com/questions/30431491/when-do-python-cached-bytecode-pyc-files-get-updated

[^1_27]: https://pypi.org/project/pytest-remove-stale-bytecode/

[^1_28]: https://dev.to/umr55766/how-to-create-an-automated-code-cleaner-with-pre-commit-hook-3j85

[^1_29]: https://www.youtube.com/watch?v=xd9aY0GP0Xo

[^1_30]: https://fedoraproject.org/wiki/Changes/Python_Optional_Bytecode_Cache

[^1_31]: https://www.pythonmorsels.com/modules-are-cached/

[^1_32]: https://discuss.python.org/t/python-pyc-files-in-a-docker-image/26816

[^1_33]: https://aleksac.me/blog/dont-use-pythondontwritebytecode-in-your-dockerfiles/

[^1_34]: https://drdroid.io/framework-diagnosis-knowledge/python-flask-flask-caching--stale-data

[^1_35]: https://www.pythonanywhere.com/forums/topic/29967/

[^1_36]: https://www.reddit.com/r/git/comments/iu75kx/help_with_gitignore_directory_pycache/

[^1_37]: https://cbtw.tech/insights/green-coding-with-python-sustainable-practices-for-developers

[^1_38]: https://discuss.python.org/t/add-gitignore-to-pycache-directories/75635

[^1_39]: https://stackoverflow.com/questions/16869024/what-is-pycache

[^1_40]: https://github.com/martinohanlon/flightlight/issues/1

[^1_41]: https://discuss.python.org/t/pycache-creation-or-inhibition-vary-by-install-location/80476

[^1_42]: https://stackoverflow.com/questions/3719243/best-practices-for-adding-gitignore-file-for-python-projects

[^1_43]: https://community.vercel.com/t/python-package-caching-strategy-for-faster-deployments/2808

[^1_44]: https://github.com/denoland/deno/issues/3335

[^1_45]: https://github.com/ipython/ipython/issues/11004

[^1_46]: https://github.com/Bahus/easy_cache

[^1_47]: https://www.dummies.com/article/technology/programming-web-design/python/using-python-environment-variables-advantage-250346/

[^1_48]: https://python.useinstructor.com/blog/2023/11/26/python-caching-llm-optimization/

[^1_49]: https://dev.to/ferdinandodhiambo/cache-invalidation-the-silent-performance-killer-1fl8

[^1_50]: https://www.ipway.com/blog/python-cache-accelerate-your-code/

[^1_51]: https://github.com/jordansissel/fpm/issues/1690

[^1_52]: https://www.datacamp.com/tutorial/python-cache-introduction

[^1_53]: https://stackoverflow.com/questions/15839555/when-are-pyc-files-refreshed

[^1_54]: https://iproyal.com/blog/python-cache-basics/

[^1_55]: https://docs.python.org/3.11/whatsnew/changelog.html

[^1_56]: https://www.techgrind.io/explain/what-is-the-best-way-to-clear-out-all-the-__pycache__-folders-and-pycpyo-files-from-a-python3-project

[^1_57]: https://github.com/bazel-contrib/rules_python/issues/1761

[^1_58]: https://discuss.python.org/t/pysource-file-layout-for-installed-modules/14594

[^1_59]: https://github.com/python/cpython/issues/106911

[^1_60]: https://www.reddit.com/r/learnpython/comments/6c1t5m/sometimes_i_see_python_creates_pycache_folder/

[^1_61]: https://rohitkrsingh.hashnode.dev/demystifying-pycache-in-python/rss.xml

[^1_62]: https://pre-commit.com

[^1_63]: https://github.com/Miserlou/Zappa/issues/1356

[^1_64]: https://codemia.io/knowledge-hub/path/if_python_is_interpreted_what_are_pyc_files

[^1_65]: https://www.reddit.com/r/Python/comments/1747md/git_tip_remove_pyc_files_automatically/

[^1_66]: https://www.reddit.com/r/learnpython/comments/5rjr4c/does_compiled_python_pyc_include_packages/

[^1_67]: https://github.com/pre-commit/pre-commit/issues/2286

[^1_68]: https://discuss.python.org/t/change-pyc-file-format-to-record-more-accurate-timestamp-and-other-information/57815

[^1_69]: https://packages.msys2.org/packages/mingw-w64-x86_64-python-pre-commit

[^1_70]: https://news.ycombinator.com/item?id=23366871

[^1_71]: https://blog.phylum.io/compiled-python-files/

