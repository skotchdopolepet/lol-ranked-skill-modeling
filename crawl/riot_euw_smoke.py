import main_dataset as _impl

# Re-export all implementation symbols (including single-underscore helpers)
# so existing imports from riot_euw_smoke keep working.
for _name, _value in vars(_impl).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


if __name__ == "__main__":
    _impl.main()
