





# Evaluation on the GenEval

Evaluation on the GenEval benchmark. * result is with prompt rewriting.

| Methods   | # Params | Single Obj. | Two Obj. | Count. | Colors | Position | Color Attri. | Overall $\uparrow$ |
| --------- | -------- | ----------- | -------- | ------ | ------ | -------- | ------------ | ------------------ |
| Infinity  | 2.20B    | 0.9938      | 0.7879   | 0.6688 | 0.8271 | 0.2525   | 0.6100       | 0.6900             |
| Infinity* | 2.20B    | 1.0000      | 0.8586   | 0.6594 | 0.8537 | 0.4450   | 0.5675       | 0.7307             |
| +ScaleKV* | 2.20B    | 1.0000      | 0.8510   | 0.6750 | 0.8484 | 0.4375   | 0.5400       | 0.7253             |
| FastVAR*  | 2.20B    | 1.0000      | 0.8283   | 0.6406 | 0.8271 | 0.4350   | 0.4950       | 0.7043             |



Infinity: 32903MiB / 40536MiB

ScaleKV: 22705MiB / 40536MiB


# How to eval

```python
pip install -U openmim
mim install mmengine mmcv-full==1.7.2
pip install mmdet==2.28.2 pytorch_lightning clip_benchmark open-clip-torch==2.20.0
# pip install -U diffusers
apt install libgl1
# pip install openai
# pip install httpx==0.20.0
```

如果在执行 `mim install mmengine mmcv-full==1.7.2` 时报错如下，请卸载并重新安装 `setuptools`
```python
Traceback (most recent call last):
  File ".../anaconda3/envs/torch260/bin/mim", line 5, in <module>
    from mim.cli import cli
  File ".../anaconda3/envs/torch260/lib/python3.12/site-packages/mim/__init__.py", line 10, in <module>
    import setuptools  # noqa: F401
    ^^^^^^^^^^^^^^^^^
  File ".../anaconda3/envs/torch260/lib/python3.12/site-packages/setuptools/__init__.py", line 16, in <module>
    import setuptools.version
  File ".../anaconda3/envs/torch260/lib/python3.12/site-packages/setuptools/version.py", line 1, in <module>
    import pkg_resources
  File ".../anaconda3/envs/torch260/lib/python3.12/site-packages/pkg_resources/__init__.py", line 2172, in <module>
    register_finder(pkgutil.ImpImporter, find_on_path)
                    ^^^^^^^^^^^^^^^^^^^
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'?

# reinstall setuptools
pip uninstall -y setuptools
pip install setuptools
```