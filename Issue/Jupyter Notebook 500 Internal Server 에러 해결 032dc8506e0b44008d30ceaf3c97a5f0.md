# Jupyter Notebook 500:Internal Server 에러 해결

👋 Github를 사용하게 되면서 코드파일의 로컬 위치를 옮겼더니 500:Internal Server 에러가 떴다.

# 쉬운 방법

```
pip install --upgrade jupyter
```

# 그래도 안됐을 때

```
pip install --upgrade jupyterhub
pip install --upgrade --user nbconvert
conda install nbconvert==5.4.1
pip install --upgrade jupyter
```

# 참고자료

[스택오버플로우](https://stackoverflow.com/questions/36851746/jupyter-notebook-500-internal-server-error)