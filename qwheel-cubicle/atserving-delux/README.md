## atserving_delux 

并发调用atserving线上服务，目前支持`Linux`, `macOS`。

## Requirements

* 对应操作系统
* Python 2.7
* 没了

## Usage

### download exec

可以在下载完成后重命名为 atserving_delux，并**确认文件的可执行权限**

#### v1.1 

> update 2018-08-01

* [linux](http://pbmt9e0id.bkt.clouddn.com/atserving_delux/atserving_delux_linux_v1.1)
* [mac](http://pbmt9e0id.bkt.clouddn.com/atserving_delux/atserving_delux_mac_v1.1)

#### v1.0 

> update 2018-07-31

* [linux](http://pbmt9e0id.bkt.clouddn.com/atserving_delux/atserving_delux_linux_v1.0)

### rock 'n roll

0. 查看使用帮助 `-h` 

1. 在`atserving_delux`同级目录下创建配置文件`atserving_delux.conf`，如果是其他路径和名字，但是需要加进`--cfg`参数。
   配置文件参考模板：

   ```
   # configuration file for atserving_delux.py
   
   [keys]
   ak = -
   sk = -
   
   [params]
   host = argus.atlab.ai
   # set as image or video
   data_type = video
   
   # post qpulp image
   # query = /v1/pulp
   
   # post qpulp video
   query = /v1/video/demovideoid
   async = False
   
   # video api
   [vframe]
   interval = 5
   mode = 0
   
   [ops]
   # split with comma
   op = pulp,terror,politician
   ```

2. 调用示例：

   ```
   $ ./atserving_delux -s --url 'http://pak58nghz.bkt.clouddn.com/pulp_samples/youtube_sexy_1.mp4'
   
   ================================================================================
   Arguments submitted:
   cfg                 = ./atserving_delux.conf
   help                = False
   single-mode         = True
   url                 = http://pak58nghz.bkt.clouddn.com/pulp_samples/youtube_sexy_1.mp4
   version             = False
   ================================================================================
   => Start processing...
   => ...Configuration file loaded
   => Host: argus.atlab.ai
      Post: /v1/video/demovideoid
      Data uri: http://pak58nghz.bkt.clouddn.com/pulp_samples/youtube_sexy_1.mp4
   => Posting...
   => Response time: 2.335s
   => Result:
   {u'politician': {u'segments': None},
    u'pulp': {u'labels': [{u'label': u'2', u'score': 0.9909987}],
              u'segments': [{u'cuts': [{u'offset': 0,
                                        u'result': {u'label': 2,
                                                    u'review': False,
                                                    u'score': 0.79848456}},
                                       {u'offset': 5000,
                                        u'result': {u'label': 2,
                                                    u'review': False,
                                                    u'score': 0.9909987}},
                                       {u'offset': 10000,
                                        u'result': {u'label': 2,
                                                    u'review': False,
                                                    u'score': 0.9861078}}],
                             u'labels': [{u'label': u'2',
                                          u'score': 0.9909987}],
                             u'offset_begin': 0,
                             u'offset_end': 10000}]},
    u'terror': {u'labels': [{u'label': u'0', u'score': 0.9845994}],
                u'segments': [{u'cuts': [{u'offset': 0,
                                          u'result': {u'label': 0,
                                                      u'review': False,
                                                      u'score': 0.9845994}},
                                         {u'offset': 5000,
                                          u'result': {u'label': 0,
                                                      u'review': False,
                                                      u'score': 0.9831457}},
                                         {u'offset': 10000,
                                          u'result': {u'label': 0,
                                                      u'review': False,
                                                      u'score': 0.8602738}}],
                               u'labels': [{u'label': u'0',
                                            u'score': 0.9845994}],
                               u'offset_begin': 0,
                               u'offset_end': 10000}]}}
   => ...Done
   ```

   

3. 

