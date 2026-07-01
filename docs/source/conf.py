import os
import sys

# pyena 소스를 import 경로에 추가 (docs/source 기준 ../../src)
sys.path.insert(0, os.path.abspath("../../src"))

project = "pyena"
copyright = "2026, JongHwi Song"
author = "JongHwi Song"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",       # docstring에서 API 문서 자동 생성
    "sphinx.ext.napoleon",      # NumPy/Google 스타일 docstring 파싱
    "sphinx.ext.viewcode",      # 소스 코드 링크
    "sphinx.ext.intersphinx",   # numpy/pandas/sklearn 문서 상호참조
    "myst_parser",              # Markdown 지원 (튜토리얼용)
]

# NumPy 스타일 docstring
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# autodoc 설정
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

# 외부 라이브러리 상호참조
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

# Read the Docs 테마
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Markdown + reStructuredText 둘 다 허용
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
