# 가상환경 Computer Setting

👋 Anaconda 가상환경에 Gpu 연결 확인 - CUDA, Tensorflow, pytorch, transformers, sklearn 설치 

# 구축하려는 컴퓨터 환경

| Window | Windows 10 Education 64비트 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Ti |
| NVIDIA DRIVER | 531.61 |
| CUDA Toolkit version | 11.8 |
| cuDNN | cudnn-windows-x86_64-8.9.0.131_cuda11-archive |
| Python | 3.9.0 |
| Tensorflow | 2.10.0  |
| pyTorch | 2.0.0 |
| transformers | 4.24.0 |

# 0. 호환성 확인하기

현재 사용중인 노트북에서 RTX 4070 Ti를 사용중이라서 compute capability 8.9 입니다.

[CUDA](https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications)

![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled.png)

앞에서 언급한 링크에서 다시 확인해보면 compute capability 8.9 인경우 CUDA SDK 11.8 이후 버전을 사용할 수 있습니다.

![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%201.png)

pytorch의 경우 CUDA가 11.8까지 지원되므로, 최종적으로 11.8 버전을 다운 받아야 하는 것을 확인

[PyTorch](https://pytorch.org/get-started/locally/#windows-anaconda)

![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%202.png)

# 1. Visual Studio 설치

[CUDA Toolkit 가이드](https://developer.nvidia.com/cuda-toolkit-archive)에서 다운받을 CUDA 버전의 온라인 Doc를 클릭 

→ `Installation Guides > Installation Guide Windows` 을 차례대로 클릭해보자. 

아래 붉은색으로 표시한 부분에서 Cuda 11.8이 윈도우 10에서 지원한다는 사실과 
컴파일러 도구로 Visual Studio 2022 17.0 버전을 설치해야 한다는 것을 알게 되었다.

![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%203.png)

![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%204.png)

- ? tensorflow [**버전 확인**](https://www.tensorflow.org/install/source_windows#tested_build_configurations)을 해보면, MSVC 버전이 2019인 것을 확인할 수 있다. 
이건 최소 버전인 듯 하다. visual studio 2022 / cuDNN 8.9 / CUDA 11.8 설치했는데 잘 됐다.

![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%205.png)

1. [Visual Studio 2022 Community](https://visualstudio.microsoft.com/ko/downloads/?ranMID=24542&ranEAID=je6NUbpObpQ&ranSiteID=je6NUbpObpQ-AxduGcXjHdb3KUKf82TGeg&epi=je6NUbpObpQ-AxduGcXjHdb3KUKf82TGeg&irgwc=1&OCID=AID2200057_aff_7593_1243925&tduid=%28ir__il1scyvf0kkfblmysl69hdao6e2x6p3g1dfc2saw00%29%287593%29%281243925%29%28je6NUbpObpQ-AxduGcXjHdb3KUKf82TGeg%29%28%29&irclickid=_il1scyvf0kkfblmysl69hdao6e2x6p3g1dfc2saw00) 버전을 다운
2. 다운로드 한 파일을 더블클릭하여 설치 중 아래 화면과 같이 `C++를 사용한 데스크톱 개발`
에 클릭한 후 설치
    
    ![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%206.png)
    

# 2. GPU 드라이버 설치

- 드라이버 다운로드

[](https://www.nvidia.com/download/index.aspx?lang=kr)

![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%207.png)

1. 검색을 누르고 설치 
2. *GeForce Experience는 딱히 필요 없습니다*
3. 사용자 정의로 설치 후에 Experience가 체크 해제된 것을 확인하고 설치 완료
4. 설치 확인
    1. cmd에 가서 터미널에 명령어 입력 
        
        ```
        nvidia-smi
        ```
        
    
     b. 정상적으로 NVidia 그래픽 드라이버가 설치 되었는지, 추천 CUDA Version은 무엇인지 확인
    
        (여기서 CUDA version은 **추천**일 뿐, 1번에서 확인한 CUDA version으로 다운 받아야 합니다.)
    
    ![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%208.png)
    

# 3. CUDA Toolkit 11.8 다운

[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)

![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%209.png)

1. 설치 시작(라이선스 동의)
2. 사용자 정의 설치
3. *NVidia GeForce Experience 체크 해제*
4. NEXT를 눌러 끝까지 **설치를 완료**합니다.

# 4. cuDNN SDK 설치

멤버십이 요구 되므로, 회원가입을 안하신 분들은 회원가입 후 **로그인을 진행**합니다.

[NVIDIA CUDA Deep Neural Network (cuDNN)](https://developer.nvidia.com/cudnn)

1. *자신이 설치한 **CUDA 버전에 맞는 cuDNN을 선택**하여 다운로드 합니다.*
    
    ![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%2010.png)
    
2. *다운로드 받은 zip 파일의 압축을 해제 합니다.*

1. *안에 있는 파일을 CUDA Computing Toolkit 에 복사*
    
    `cuda\bin` 폴더 안의 모든 파일은 => `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
    
    `cuda\include` 폴더 안의 모든 파일은 => `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include`
    
    `cuda\lib` 폴더 안의 모든 파일은 => `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib`
    
2. `Window + R`  키를 누른 후 “control sysdm.cpl”을 실행합니다.
    1. **고급 탭 - 환경변수를 클릭**
    2. `CUDA_PATH` 가 다음과 같이 잘 등록되어 있는지 확인합니다.
    
    ![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%2011.png)
    
    ![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%2012.png)
    
    # 5. Python, Tensorflow 설치
    
    - [tensorflow 호환성 확인](https://www.tensorflow.org/install?hl=ko)
    
    ![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%2013.png)
    
    - "pip install tensorflow"은 tensorflow 버전 안맞게 설치됨
    - 따라서 **[버전 확인](https://www.tensorflow.org/install/source_windows#tested_build_configurations)** 필요
    
    ![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%205.png)
    
    cuDNN과  CUDA의 버전은 현재 설치한 것의 아래이기만 하면 되기 때문에
    tensorflow는 가장 최신인 2.10.0 설치할 것 (Python 버전은 3.9)
    
    ```
    * Anaconda에 입력
    conda create -n tensorflow2_py39 python=3.9   # nlp라는 이름의 가상환경 생성
    conda activate tensorflow2_py39               # nlp 가상환경에 들어가기
    pip install jupyter notebook      # 가상환경에 jupyter notebook 설치
    
    # 가상환경에 kernel 연결
    python -m ipykernel install --user --name tensorflow2_py39 --display-name tensorflow2_py39 
    
    pip install tensorflow-gpu==2.10.0    # 가상환경에 tensorflow 설치
    jupyter notebook                  # 주피터 노트북 실행
    ```
    
    tensorflow와 gpu 연결 확인 : device에서 gpu가 뜨면 된다.
    
    ```python
    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()
    ```
    
    ![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%2014.png)
    
    # 6. Pytorch 다운
    
    [pytorch 설치 페이지](https://pytorch.org/get-started/locally/#windows-anaconda)에 가서 CUDA 버전에 맞는 명령어를 받아온다.
    
    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
    
    ![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%2015.png)
    
    설치가 잘 되었는지 확인
    
    ```python
    import torch
    
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.cuda.get_device_name(0))
    ```
    
    ![Untitled](%E1%84%80%E1%85%A1%E1%84%89%E1%85%A1%E1%86%BC%E1%84%92%E1%85%AA%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%20Computer%20Setting%203c42a49bf39943fbb3bb5663021a32f9/Untitled%2016.png)
    
    # 7. 기타 라이브러리 설치
    
    [머신러닝을 위한 파이썬의 도구들](https://theorydb.github.io/review/2019/06/05/review-book-intro-ml-py/#%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%84-%EC%9C%84%ED%95%9C-%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9D%98-%EB%8F%84%EA%B5%AC%EB%93%A4scikit-learn-%EB%93%B1)
    
    ```
    conda install -n tensorflow2_py39 scikit-learn matplotlib spyder pandas seaborn transformers
    ```
    
    # 참고자료
    
    - [cuda, cudnn, tensorflow 설치 버전확인](https://webnautes.tistory.com/1454)
    - [전체적으로 설치하는 방법](https://teddylee777.github.io/colab/tensorflow-gpu-install-windows)
    - [딥러닝 개발 환경 구축 한방에 끝내기](https://theorydb.github.io/dev/2020/02/14/dev-dl-setting-local-python)