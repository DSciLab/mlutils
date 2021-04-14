from distutils.core import setup
import datetime


def gen_code():
    d = datetime.datetime.now()
    date_str = d.strftime('%Y%m%d%H%M%S')
    return f'dev{date_str}'


__version__ = f'0.0.1-{gen_code()}'


setup(name='mlutils_dist',
      version=__version__,
      description='Machine learn utils',
      author='tor4z',
      author_email='vwenjie@hotmail.com',
      install_requires=[
            'torch',
            'numpy',
            'tqdm',
            'scikit-learn',
            'visdom',
            'imageio',
            'libtiff',
            'opencv-python'
      ],
      packages=['mlutils', 'mlutils/thirdparty'],
     )
