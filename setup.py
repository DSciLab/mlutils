from distutils.core import setup
import mlutils


setup(name='mlutils_dist',
      version=mlutils.__version__,
      description='Machine learn utils',
      author='tor4z',
      author_email='vwenjie@hotmail.com',
      install_requires=[
            'numpy',
            'tqdm',
            'scikit-learn',
            'visdom',
            'torch',
            'imageio',
            'libtiff',
            'opencv-python'
      ],
      packages=['mlutils', 'mlutils/thirdparty'],
     )
