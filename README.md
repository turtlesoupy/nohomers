=This Fucked Up Homer Does Not Exist=

Source code for https://www.thisfuckeduphomerdoesnotexist.com/ - a website
dedicated to failure in AI generated Simpsons characters. 

This code requires use of a training set, and trained generative model using
@lucidrains https://github.com/lucidrains/lightweight-gan

My production run used the following command line training params:
```
lightweight_gan --data /mnt/evo/projects/metapedia/tmp/google_images_scrape/simpsons_large_cleaned_nobackground_1024 \
--name simpsons_large_cleaned_nobackground_1024_augall03_sle_res64 --aug-prob 0.3 --image-size 1024 --batch-size 12 \
--gradient-accumulate-every 3 --attn-res-layers "[32,64]"  --sle-spatial --aug-types "[translation,cutout,color]" \
--save-every 2500 --evaluate-every 2500
```