#include "PluginProcessor.h"
#include "PluginEditor.h"


//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p)
{
    juce::ignoreUnused (processorRef);
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.

    // Main window.
    setSize(M_WINDOW_WIDTH, M_WINDOW_HEIGHT);
    setResizable(M_IS_WIDTH_RESIZABLE, M_IS_HEIGHT_RESIZABLE);
    
    // Lambda function for allowing Sun object to generate random sounds.
    mSun.getNewSample = [&]() {
        processorRef.generator.generateSample(
            processorRef.generator.generateLatents()
        );
    };

    /*
    mPlanet.setDiameter(M_DEFAULT_PLANET_DIAMETER);
    mPlanet.setWindowBoundary(M_WINDOW_WIDTH, M_WINDOW_HEIGHT);
    */

    addAndMakeVisible(mMap);
    addAndMakeVisible(mSun);
    //addAndMakeVisible(mPlanet);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g)
{
}

void AudioPluginAudioProcessorEditor::resized()
{
    auto r = getLocalBounds();
    auto mapArea = r;
    mMap.setBounds(mapArea);
    
    // Sets the sun object in center of the window.
    mSun.setBounds(
        (getWidth() - mSun.getDiameter()) / 2,
        (getHeight() - mSun.getDiameter()) / 2,
        mSun.getDiameter(),
        mSun.getDiameter()
    );

    //mPlanet.setBounds(mapArea);
}
