## domain_pop 

根据域名，调用工具获取对应空间名，目前支持`macOS`。

## Requirements

* 对应操作系统
* Python 2.7
* qshell
* 没了

## Usage

### download exec

可以在下载完成后重命名为 domain_pop，并**确认文件的可执行权限**

#### v1.0 

> update 2018-07-25

* [mac](http://pbmt9e0id.bkt.clouddn.com/domain_pop/domain_pop_mac_v1.0)

### rock 'n roll

0. 查看使用帮助 `-h` 

1. 准备工作: 下载对应版本的[qshell](https://developer.qiniu.com/kodo/tools/1302/qshell)，放在domain_pop同目录下，或者在运行时写到`--qshell-path`参数中。

2. 首次运行需要登录，并将ak/sk填到`--login`参数中，以逗号分隔。
```
$ python domain_pop domain-test.com --login your_ak,your_sk 

================================================================================
Arguments submitted:
clear               = False
help                = False
login               = your_ak,your_sk 
qshell-path         = ./qshell
update              = False
version             = False
<domain>            = domain-test.com
================================================================================
=> start querying...
WARNING: Only bucket with one single domain could be catched for now
=> checking essential tools...
=> qshell version: QShell/v2.0.6 (darwin; amd64; go1.7)
=> logging...
=> updating cache...
=> searching for domain:
domain-test.com
=> bucket name found:
bucket-test
=> ...done
```

3. 已经登录过`qshell`之后，查询时不需要再加`--login`参数。 

4. `--login`参数会在登录后自动进行`-u|--update`，之后再进行查询不再需要单独加`-u`参数来更新缓存。

5. 如果账号下创建了新的空间，则需要加`-u`来更新缓存，否则查询不到新建的空间。

6. 正常查询不需要加任何额外参数。
```
$ python domain_pop.py domain-test.com 

================================================================================
Arguments submitted:
clear               = False
help                = False
login               = None
qshell-path         = ./qshell 
update              = False
version             = False
<domain>            = domain-test.com 
================================================================================
=> start querying...
WARNING: Only bucket with one single domain could be catched for now
=> checking dependencies...
qshell version: QShell/v2.0.6 (darwin; amd64; go1.7)
=> searching for domain:
domain-test.com
=> bucket name found:
bucket-test
=> ...done
```

