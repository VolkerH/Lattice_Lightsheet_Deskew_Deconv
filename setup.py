from setuptools import setup

setup(name='lls_dd',
      version='2019.3.a1',
      description='lattice lightsheet deskewing and deconvolution',
      url='https://github.com/VolkerH/Lattice_Lightsheet_Deskew_Deconv',
      author='Volker Hilsenstein',
      author_email='volker.hilsenstein@monash.edu',
      license='BSD-3',
      packages=['lls_dd'],
      zip_safe=False,
      entry_points='''
            [console_scripts]
            lls_dd=lls_dd.cmdline:cli
      ''',
      )
