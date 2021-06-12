#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor(AudioPluginAudioProcessor& p)
    :   AudioProcessorEditor(&p),
        processorRef(p),
        m_Map(processorRef.m_AudioContainer, processorRef.m_Parameters){
    Logger::writeToLog("Editor created.");
    
    addAndMakeVisible(m_Map);

    // Main window.
    setSize(Variables::WINDOW_WIDTH, Variables::WINDOW_HEIGHT);
    setResizable(Variables::IS_WIDTH_RESIZABLE, Variables::IS_HEIGHT_RESIZABLE);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor(){Logger::writeToLog("Editor destroyed.");}

//------------------------------------------------------------//
// View methods.

void AudioPluginAudioProcessorEditor::paint(juce::Graphics& g){juce::ignoreUnused(g);}

void AudioPluginAudioProcessorEditor::resized(){
    auto r = getLocalBounds();
    auto mapArea = r;
    m_Map.setBounds(mapArea);
}
