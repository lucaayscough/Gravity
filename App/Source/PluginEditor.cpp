#include "Headers.h"


//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p)
{
    juce::ignoreUnused (processorRef);

    // Main window.
    setSize(M_WINDOW_WIDTH, M_WINDOW_HEIGHT);
    setResizable(M_IS_WIDTH_RESIZABLE, M_IS_HEIGHT_RESIZABLE);
    
    // Make map visible.
    addAndMakeVisible(m_Map);

    // Pass the map a pointer to the generator class.
    m_Map.setGeneratorAccess(&processorRef.generator);

    // Create sun inside the map.
    m_Map.createSun();
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor(){}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g){}

void AudioPluginAudioProcessorEditor::resized(){
    auto r = getLocalBounds();
    auto mapArea = r;
    m_Map.setBounds(mapArea);
}
