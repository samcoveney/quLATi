# TODO


## Multi-output moodel with heteroscedastic noise

1) allow nuggets to be fixed for gpmi_mo models
2) allow heteroscedastic observation noise for gpmi_mo models

Supplying _correlated_ heteroscedastic observation noise (i.e. non-diagonal noise matrix) should work in theory, as the Woodbury inversion formula should still apply. However, it may cause numerically difficulties.


## Between-outputs correlations

For training the gpmi_mo model, it may be sensible to supply an initial guess for the between-outputs correlation coefficients. This guess could be derived from the correlation matrix of the observed outputs.

