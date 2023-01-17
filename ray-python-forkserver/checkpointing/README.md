
```
sudo ../criu/criu/criu dump -t 15209 --shell-job
sudo ../criu/criu/criu restore --shell-job
```

```
# Empty python process
$ ls -alh
total 3.3M
drwxrwxr-x 2 ubuntu ubuntu 4.0K Jan 17 01:35 .
drwxrwxr-x 4 ubuntu ubuntu 4.0K Jan 17 01:44 ..
-rw-r--r-- 1 root   root   2.5K Jan 17 01:34 core-15209.img
-rw-r--r-- 1 root   root     44 Jan 17 01:34 fdinfo-2.img
-rw-r--r-- 1 root   root   1.1K Jan 17 01:34 files.img
-rw-r--r-- 1 root   root     18 Jan 17 01:34 fs-15209.img
-rw-r--r-- 1 root   root     36 Jan 17 01:34 ids-15209.img
-rw-r--r-- 1 root   root     46 Jan 17 01:34 inventory.img
-rw-r--r-- 1 root   root   1.7K Jan 17 01:34 mm-15209.img
-rw-r--r-- 1 root   root    768 Jan 17 01:34 pagemap-15209.img
-rw-r--r-- 1 root   root   3.2M Jan 17 01:34 pages-1.img
-rw-r--r-- 1 root   root     26 Jan 17 01:34 pstree.img
-rw-r--r-- 1 root   root     12 Jan 17 01:34 seccomp.img
-rw-r--r-- 1 root   root     47 Jan 17 01:34 stats-dump
-rw-r--r-- 1 root   root     26 Jan 17 01:43 stats-restore
-rw-r--r-- 1 root   root     34 Jan 17 01:34 timens-0.img
-rw-r--r-- 1 root   root    204 Jan 17 01:34 tty-info.img
```

```
# Python + TF imported
$ ls -alh
total 147M
drwxrwxr-x 2 ubuntu ubuntu 4.0K Jan 17 01:48 .
drwxrwxr-x 4 ubuntu ubuntu 4.0K Jan 17 01:48 ..
-rw-r--r-- 1 root   root   2.6K Jan 17 01:48 core-16812.img
-rw-r--r-- 1 root   root   1.4K Jan 17 01:48 core-16813.img
-rw-r--r-- 1 root   root   1.4K Jan 17 01:48 core-16814.img
-rw-r--r-- 1 root   root   1.4K Jan 17 01:48 core-16815.img
-rw-r--r-- 1 root   root   1.4K Jan 17 01:48 core-16816.img
-rw-r--r-- 1 root   root   1.4K Jan 17 01:48 core-16817.img
-rw-r--r-- 1 root   root   1.4K Jan 17 01:48 core-16818.img
-rw-r--r-- 1 root   root   1.4K Jan 17 01:48 core-16819.img
-rw-r--r-- 1 root   root   1.4K Jan 17 01:48 core-16820.img
-rw-r--r-- 1 root   root   1.4K Jan 17 01:48 core-16821.img
-rw-r--r-- 1 root   root   1.4K Jan 17 01:48 core-16822.img
-rw-r--r-- 1 root   root   1.4K Jan 17 01:48 core-16823.img
-rw-r--r-- 1 root   root   1.4K Jan 17 01:48 core-16824.img
-rw-r--r-- 1 root   root   1.4K Jan 17 01:48 core-16825.img
-rw-r--r-- 1 root   root   1.4K Jan 17 01:48 core-16826.img
-rw-r--r-- 1 root   root   1.4K Jan 17 01:48 core-16827.img
-rw-r--r-- 1 root   root     47 Jan 17 01:48 fdinfo-2.img
-rw-r--r-- 1 root   root    20K Jan 17 01:48 files.img
-rw-r--r-- 1 root   root     20 Jan 17 01:48 fs-16812.img
-rw-r--r-- 1 root   root     36 Jan 17 01:48 ids-16812.img
-rw-r--r-- 1 root   root     46 Jan 17 01:48 inventory.img
-rw-r--r-- 1 root   root    28K Jan 17 01:48 mm-16812.img
-rw-r--r-- 1 root   root   6.2K Jan 17 01:48 pagemap-16812.img
-rw-r--r-- 1 root   root   147M Jan 17 01:48 pages-1.img
-rw-r--r-- 1 root   root     89 Jan 17 01:48 pstree.img
-rw-r--r-- 1 root   root     12 Jan 17 01:48 seccomp.img
-rw-r--r-- 1 root   root     53 Jan 17 01:48 stats-dump
-rw-r--r-- 1 root   root     28 Jan 17 02:04 stats-restore
-rw-r--r-- 1 root   root     34 Jan 17 01:48 timens-0.img
-rw-r--r-- 1 root   root    193 Jan 17 01:48 tty-info.img
```
