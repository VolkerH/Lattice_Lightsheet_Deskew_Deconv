# Folder structure and naming for experiments

This is what the folder and file structure for a lattice light sheet experiment at Monash Micro Imaging look like:

```
volker@LAPTOP-HDUOL5Q3:/mnt/c/Users/Volker/Data$ tree Experiment_testing_stacks/           
Experiment_testing_stacks/                                                                 
├── PSF                                                                                    
│   ├── 488                                                                                
│   │   ├── Galvo                                                                          
│   │   │   ├── 488 PSF galvo_Settings.txt                                                 
│   │   │   └── 488 PSF galvo_ch0_stack0000_488nm_0000000msec_0008687820msecAbs.tif        
│   │   └── Scan                                                                           
│   │       ├── 488 PSF scan_Settings.txt                                                  
│   │       └── 488 PSF scan_ch0_stack0000_488nm_0000000msec_0008654015msecAbs.tif         
│   ├── 560                                                                                
│   │   ├── Galvo                                                                          
│   │   │   ├── 560 PSF galvo_Settings.txt                                                 
│   │   │   └── 560 PSF galvo_ch0_stack0000_560nm_0000000msec_0007423250msecAbs.tif        
│   │   └── Scan                                                                           
│   │       ├── 560 PSF scan_Settings.txt                                                  
│   │       └── 560 PSF scan_ch0_stack0000_560nm_0000000msec_0007469344msecAbs.tif         
│   └── 642                                                                                
│       ├── Galvo                                                                          
│       │   ├── 642 PSF galvo_Settings.txt                                                 
│       │   └── 642 PSF galvo_ch0_stack0000_642nm_0000000msec_0007865898msecAbs.tif        
│       └── Scan                                                                           
│           ├── 642 PSF scan_Settings.txt                                                  
│           └── 642 PSF scan_ch0_stack0000_642nm_0000000msec_0007895353msecAbs.tif         
├── Stacks                                                                                 
│   ├── stack_1                                                                            
│   │   ├── 488_300mW_642_350mW_Settings.txt                                               
│   │   ├── 488_300mW_642_350mW_ch0_stack0000_488nm_0000000msec_0013822133msecAbs.tif      
│   │   ├── 488_300mW_642_350mW_ch0_stack0001_488nm_0008450msec_0013830583msecAbs.tif      
│   │   ├── 488_300mW_642_350mW_ch1_stack0000_642nm_0000000msec_0013822133msecAbs.tif      
│   │   └── 488_300mW_642_350mW_ch1_stack0001_642nm_0008450msec_0013830583msecAbs.tif      
│   └── stack_2                                                                            
│       ├── 488_300mW_642_350mW_Settings.txt                                               
│       ├── 488_300mW_642_350mW_ch0_stack0000_488nm_0000000msec_0013822133msecAbs.tif      
│       ├── 488_300mW_642_350mW_ch0_stack0001_488nm_0008450msec_0013830583msecAbs.tif      
│       ├── 488_300mW_642_350mW_ch1_stack0000_642nm_0000000msec_0013822133msecAbs.tif      
│       └── 488_300mW_642_350mW_ch1_stack0001_642nm_0008450msec_0013830583msecAbs.tif      
└── System                                                                                 
    ├── 488nm 10um 1b 0.550 0.440 1.092 0.050 x-3.00 y-1.50 -1.0 deg.bmp                   
    ├── 488nm 10um 42b 0.550 0.440 1.092 0.180 x-3.00 y-1.50 -1.0 deg.bmp                  
    ├── 560nm 10um 1b 0.550 0.440 1.258 0.050 x-3.00 y-1.50 0.0 deg.bmp                    
    ├── 560nm 10um 38b 0.550 0.440 1.258 0.180 x-3.00 y-1.50 -1.0 deg.bmp                  
    ├── 642nm 10um 1b 0.550 0.440 1.422 0.050 x-3.00 y-1.50 -1.0 deg.bmp                   
    ├── 642nm 10um 34b 0.550 0.440 1.422 0.200 x-3.00 y-1.50 -1.0 deg.bmp                  
    ├── X tilt.JPG                                                                                                                                              
    └── Z tilt.JPG                                                            
                                                                                           
```

Note that is is probably quite similar to any LLS microscope running the Janelia Labview software, but other facilities might be naming/organizing their files slightly differently.

## PSF folders

The experiment subfolder `PSF` contains subfolders for each wavelength with acquisitions of bead images. There are subfolders for `Galvo`-scanned bead volumes (these are not skewed) and for z-Stage scanned (`Scan`) bead volumes. `lls_dd` is looking for the `Galvo`-scanned bead volumes only.
Each PSF acquistion subfolder must contain a `.*Settings.txt`

## Stacks

The experiment subfolder `Stacks` contains subfolders that represent one time-series acquisition each (IMHO they would more aptly be named `Acquistions`
but I didn't create the naming scheme). Each `.tif` file in these subfolders holds one raw 3D-stack. The time point and channel wavelength are encoded in the filename of each `.tif`. Each acquistion subfolder must contain a `.*Settings.txt` file.

## System 

The `System` folder is used internally and not required for `lls_dd`.