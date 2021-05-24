#include "Headers.h"


//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p)
{
    juce::ignoreUnused (processorRef);

    // Main window.
    setSize(M_WINDOW_WIDTH, M_WINDOW_HEIGHT);
    setResizable(M_IS_WIDTH_RESIZABLE, M_IS_HEIGHT_RESIZABLE);
    
    addAndMakeVisible(m_Map);
    m_Map.setGeneratorAccess(&processorRef.generator);

    Generator* genPtr = &processorRef.generator;

    // Lambda function for allowing Sun object to generate random sounds.

    /*
    mSun.getNewSample = [&]() {
        processorRef.generator.generateSample(
            processorRef.generator.generateLatents()
        );
    };
    */    
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor(){}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g){}

void AudioPluginAudioProcessorEditor::resized(){
    auto r = getLocalBounds();
    auto mapArea = r;
    m_Map.setBounds(mapArea);
}
