### After the Build Process is complete, you should see a folder called "Release" (if you built the Release config). Inside this folder, you will find a .pyd file.

Inside of this folder, create a directory structure like

folder

    /ThisFolder
        /lfda
            __init__.py
            lfda.pyd
        setup.py

And then run

```
pip install .
```



This should install lfda as a python package. You can then check

```
from lfda import lfda
```


