# Nautilus development

This directory contains the regression test and the IRAS 07546+3928 demo for
`galspec.Nautilus_Fit`.

Run the quick checks from the repository root:

```bash
python develop/dev_nautilus/test_nautilus.py
```

Run the full-spectrum demo with six likelihood workers:

```bash
python develop/dev_nautilus/demo_ir07546_nautilus.py --cores 6
```

The demo uses the full additive 4200–6800 Å model, masks 5770–6000 Å, and does
not include a multiplicative polynomial. Nautilus writes a resumable HDF5
checkpoint while running and the completed GalSpec fit is saved to FITS.
