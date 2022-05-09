### Dataset: `Messidor`

Total number of images in the dataset: 1200

Diabetic Retinopathy Grade: 
|DR Grade|Description| Number of images 
|--------|:----:|:-----------------:|
|R0     | 𝜇𝐴 = 0 & H = 0 |546 |
|R1     | (0 < 𝜇𝐴 ≤ 5) & H = 0|153 |
|R2     |(5 < 𝜇𝐴 ≤ 15) OR (0 < 𝐻 < 5) & NV = 0 |247 |
|R3     | (𝜇𝐴 ≥ 15)  OR  (H ≥ 5)  OR  (NV = 1)|254 |
|Total   |  | 1200|

| Notations  | Description   |
|:---:|:----:|
|𝜇𝐴 | Number of microaneurysms|
|H | Number of Hemorrhages|
|NV =0 | NO neovascularization|
|NV =1 | Neovascularization|

The size of the fundus image is *1440 × 960*, *2240 × 1488*, or *2304 × 1536*.

|Dimensions (H,W,C)|No. of Images|
|:-:|:-:|
|(1488, 2240, 3)| 400|
|(960, 1440, 3)| 588|
|(1536, 2304, 3) |212|
|Total|1200|

`(H,W,C)` (Height, Width, No. of Channels)

<img src="https://github.com/GSAUC3/FYP/blob/master/graphs/hist.png">

## Sample Image
<img src="https://github.com/GSAUC3/FYP/blob/master/graphs/Screenshot.png">

