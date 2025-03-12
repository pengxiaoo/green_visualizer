我有一个数据可视化的任务，需要把一个高尔夫果岭的 3d 坐标集合转化为平面图，这个平面图的需求如下：

1. 有等高线；
2. 不同高度的颜色要有渐变（比如低洼处可以用偏蓝色，高地可以用偏红色）；
3. 要有梯度箭头来标识临近区域的高低起伏（比如箭头应从高处指向低处，箭头应在整张图上均匀分布）；

3d 坐标是以 json 形式提供，格式如下：

```
{
  "features": [
    {
      "geometry": {
        "type": "Point",
        "coordinates": [
          -122.2270621429703,
          37.741962843237815,
          -0.316197061850334
        ]
      },
      "id": "Elevation"
    },
    ... more Elevation data
    {
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            -122.22717255060286,
            37.742261414180874
          ],
          ... more 2d coordinates
        ]
      },
      "id": "GreenBorder"
    }
}
```

对数据的解释：

1. features 是一个列表，里面包含多种元素，不同元素类型通过 id 区分。我们只需关注 id 为 Elevation 和 GreenBorder 的两类元素。
2. Elevation 为 3d 坐标点，包含经纬度和海拔。features 会包含多个 Elevation。这些 Elevation 各自代表果岭上的某个点的三维坐标。
3. GreenBorder 为多边形，由多个 2d 坐标点构成，组成了果岭的边界。

请基于此，选用合适的 Python 绘图工具，帮我构建一个工程。我倾向于使用 geoPandas。不过如果你知道更合适的工具，请自便
