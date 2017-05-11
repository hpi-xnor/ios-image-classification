//
//  ViewController.h
//  WhatIsThis
//
//  Created by Haoxiang Li on 1/23/16.
//  Copyright © 2016 Haoxiang Li. All rights reserved.
//

#import <UIKit/UIKit.h>
#import "c_predict_api.h"
#import <vector>
#import <AVFoundation/AVCaptureOutput.h> // Allows us to use AVCaptureVideoDataOutputSampleBufferDelegate

#define kDefaultWidth 224
#define kDefaultHeight 224
#define kDefaultChannels 3
#define kDefaultImageSize (kDefaultWidth*kDefaultHeight*kDefaultChannels)

@interface ViewController : UIViewController <AVCaptureVideoDataOutputSampleBufferDelegate>{
    
    PredictorHandle predictor;
    
    NSString *model_symbol;
    NSData *model_params;
    NSMutableArray *model_synset;
    AVCaptureSession *captureSession;
    AVCaptureDevice *videoDevice;
    NSArray *top5Labels;
}
@property (weak, nonatomic) IBOutlet UILabel *labelTop1;
@property (weak, nonatomic) IBOutlet UILabel *labelTop2;
@property (weak, nonatomic) IBOutlet UILabel *labelTop3;
@property (weak, nonatomic) IBOutlet UILabel *labelTop4;
@property (weak, nonatomic) IBOutlet UILabel *labelTop5;
@property (weak, nonatomic) IBOutlet UIImageView *imageViewPhoto;
@property (weak, nonatomic) IBOutlet UIButton *detectionButton;
@property (weak, nonatomic) IBOutlet UIImageView *imageViewCrop;
@property (weak, nonatomic) IBOutlet UILabel *labelTime;

- (IBAction)startDetectionButtonTapped:(id)sender;
- (IBAction)stopDetectionButtonTapped:(id)sender;
- (AVCaptureSession *)createCaptureSessionFor:(AVCaptureDevice *)device;
//- (UIImage *) cropCenterRect:(UIImage *)image toSize:(int)size;

@end

