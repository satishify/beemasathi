import os

# Unit tests must not write into logs/traces; the dashboard reads that directory.
os.environ["BEEMASATHI_TRACE"] = "0"
