# -*- coding = uft-8 -*-
# @Time : 2022-04-27 21:31
# @Author : yzbyx
# @File : __init__.py
# @Software : PyCharm

# 运行时路径。并非__init__.py的路径
# BASE_DIR = r"..\process-code-test\tools"
# if Path(BASE_DIR).exists():
#     sys.path.append(BASE_DIR)
# else:
#     # 尝试下探一级路径
#     sys.path.append(r"..\process-code-test\tools")

try:
    from traj_process.tools import TrackInfo as C_Info
except ImportError:
    print("ImportError: No module named 'tools.info'")
