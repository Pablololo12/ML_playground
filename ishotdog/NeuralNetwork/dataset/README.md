# Dataset
### How to create a Dataset 101.

## Web Crawler

To use the web crawler is as easy as
`python3 webcrawler.py -s [search key] -d [foldername]`.
In our case we will use:

```bash
python3 webcrawler.py -s hotdog -d hotdog
python3 webcrawler.py -s office -d office
```
This will download all the pictures and will apply all the required augmentations.


## Get from video

In the case of the video is as easy as
```python3 get_from_video.py -f [videofile] -d [foldername]```

## Requirements

* Pillow
* requests
* parsel
* opencv2