# conftest.py — Register --cases-file for whitebox pytest
def pytest_addoption(parser):
    parser.addoption("--cases-file", action="store", default="S5_mapped_cases_high.json")
