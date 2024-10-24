asr_test_images, test_images: images used to evaluate ACC and ASR, the annotation files are outside
origin_poison_images: all poisoned images
poison_images: some poisoned images randomly selected from origin_poison_images according to different poisoning rates for each training
asr_test_images_NOfilter: all 38-label images in the test set, in order to generate different filters each time in the wasr test
asr_test_images: all 38-label images in the test set are poisoned

Replace the compressed package with the real dataset compressed package