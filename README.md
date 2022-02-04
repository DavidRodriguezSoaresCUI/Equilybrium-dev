# [Equilybrium](https://github.com/DavidRodriguezSoaresCUI/Equilybrium-dev)

> WARNING: This is a fast-evolving project that offers no guarantee of behaviour stability or continued maintaining by the author. Use at your own risk !

Equilybrium is a solution that monitors your files and their backup in a flexible way for a specific use case.

For more information, read [``equilybrium-whitepaper``](https://github.com/DavidRodriguezSoaresCUI/Equilybrium-dev/blob/main/equilybrium-whitepaper.github_encoding.md).

Note: examples below call the ``Python 3 interpreter`` using command ``python``, but on your system it could be bound to ``python3``.

## Main app

```
$ python equilybrium-dev.py --help
usage: Equilybrium (dev) [-h] [--simulate | --write_report | --differential_scan]

Contains uncategorized utils

optional arguments:
  -h, --help           show this help message and exit
  --simulate           Run as read only; do not generate files
  --write_report       Only write report
  --differential_scan  Only scan files that are NOT in database. This is useful for a quick "update" when editing
                       configuration for example.
```
To run it, simply use:
```
python equilybrium-dev.py
```

### First launch

On first launch, ``Equilybrium`` generates configuration file ``equilybrium.config.ini``. Fill it before running it again.

## Tests

A script testing proper behaviour is included: ``equilybrium-test.py``. 
```
$ python equilybrium-test.py --help
usage: Equilybrium-test [-h] [--clean]

optional arguments:
  -h, --help  show this help message and exit
  --clean     Removes generated files/folders; reverses changes
```

To run it, simply use:
```
python equilybrium-test.py
```
If successful, it should end with a ``ALL TESTS FINISHED WITH NO ERROR`` message, otherwise an error will be raised. 

You can erase test files and reverse all changes with argument ``--clean``
```
python equilybrium-test.py --clean
```
