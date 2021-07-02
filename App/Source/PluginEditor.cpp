#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor(AudioPluginAudioProcessor& p)
    :   AudioProcessorEditor(&p), processorRef(p),
        m_LeftBar(m_Maps),
        m_DropShadow(Variables::TOP_BAR_SHADOW_COLOUR, 10, juce::Point<int>({0, 0})), m_DropShadower(m_DropShadow){
    Logger::writeToLog("Editor created.");

    m_Maps.ensureStorageAllocated(Variables::NUM_MAPS);

    for(int i = 0; i < Variables::NUM_MAPS; i++){
        m_Maps.add(new Map(processorRef.m_AudioContainer, processorRef.m_Parameters));
        addChildAndSetID(m_Maps[i], juce::String(i));
    }

    addAndMakeVisible(m_TopBar);
    addAndMakeVisible(m_LeftBar);

    m_TopBar.setAlwaysOnTop(true);

    m_DropShadower.setOwner(&m_TopBar);

    // Main window.
    setSize(Variables::WINDOW_WIDTH, Variables::WINDOW_HEIGHT);
    setResizable(Variables::IS_WIDTH_RESIZABLE, Variables::IS_HEIGHT_RESIZABLE);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor(){
    Logger::writeToLog("Editor destroyed.");
}

//------------------------------------------------------------//
// View methods.

void AudioPluginAudioProcessorEditor::paint(juce::Graphics& g){
    g.fillAll(Variables::EDITOR_BG_COLOUR);
}

void AudioPluginAudioProcessorEditor::resized(){
    auto r = getLocalBounds();
    
    auto top_bar = r.removeFromTop(Variables::TOP_BAR);
    auto left_bar = r.removeFromLeft(Variables::LEFT_BAR);
    auto map_area = r.withTrimmedRight(Variables::MAP_TRIM).withTrimmedBottom(Variables::MAP_TRIM);

    for(Map* map : m_Maps){
        map->setBounds(map_area);
    }

    m_LeftBar.setBounds(left_bar);
    m_TopBar.setBounds(top_bar);
}
