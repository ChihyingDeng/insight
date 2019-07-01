# Insight
1. [Structure of the code](README.md##Structure)
1. [Instruction](README.md##Instruction)

## Structure of the code

```text
├── Img
|  ├── __init__.py
|  ├── runmodel.py
|  ├── util.py
|  ├── static
|  ├── template
|  |  ├── index.html
|  |  ├── list.html
|  |  └── img.html
├── flaskapp.py
├── wsgi.py
```

## Instruction
```
python wsgi.py
```
or
```
gunicorn --bind 0.0.0.0:5000 wsgi:app
```
