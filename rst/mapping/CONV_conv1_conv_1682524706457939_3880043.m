Network conv1_conv {
Layer conv1_conv {
Type: CONV 
Stride { X: 2, Y: 2 }
Dimensions {  N: 1,  C: 3,  Y: 230,  X: 230,  K: 64,  R: 7,  S: 7,  }
Dataflow {
SpatialMap(1 ,1) N;
TemporalMap(3 ,3) R;
TemporalMap(3, 3) S;
TemporalMap(540, 540) C;
TemporalMap(1, 1) X;
TemporalMap(1, 1) Y;
Cluster(540, P);
SpatialMap(1, 1) K;
}
}
}
