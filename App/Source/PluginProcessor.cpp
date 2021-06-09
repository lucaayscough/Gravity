#include "Headers.h"


//==============================================================================
AudioPluginAudioProcessor::AudioPluginAudioProcessor()
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ),
                       parametersId("Parameters"),
                       valueTreeState(*this, nullptr, parametersId, {}),
                       parameters(valueTreeState.state)
                       {}

AudioPluginAudioProcessor::~AudioPluginAudioProcessor(){}

//==============================================================================
const juce::String AudioPluginAudioProcessor::getName() const {return JucePlugin_Name;}

bool AudioPluginAudioProcessor::acceptsMidi() const{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool AudioPluginAudioProcessor::producesMidi() const{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool AudioPluginAudioProcessor::isMidiEffect() const{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double AudioPluginAudioProcessor::getTailLengthSeconds() const {return 0.0;}
int AudioPluginAudioProcessor::getNumPrograms(){return 1;}
int AudioPluginAudioProcessor::getCurrentProgram(){return 0;}
void AudioPluginAudioProcessor::setCurrentProgram (int index){juce::ignoreUnused (index);}

const juce::String AudioPluginAudioProcessor::getProgramName (int index){
    juce::ignoreUnused (index);
    return {};
}

void AudioPluginAudioProcessor::changeProgramName (int index, const juce::String& newName){juce::ignoreUnused (index, newName);}

//==============================================================================
void AudioPluginAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // Use this method as the place to do any pre-playback
    // initialisation that you need..
    juce::ignoreUnused (sampleRate, samplesPerBlock);
}

void AudioPluginAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

bool AudioPluginAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}

void AudioPluginAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                              juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused (midiMessages);

    juce::ScopedNoDenormals noDenormals;
    int totalNumInputChannels  = getTotalNumInputChannels();
    int totalNumOutputChannels = getTotalNumOutputChannels();
    int numSamples = buffer.getNumSamples();

    while(m_AudioContainer.sampleIndex.size() <= totalNumOutputChannels){
        m_AudioContainer.sampleIndex.add(0);
    }
    
    // Clear extra channel buffers.
    for (int i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    // Check for midi data.
    for(const MidiMessageMetadata metadata : midiMessages){
        Logger::writeToLog(metadata.getMessage().getDescription());
        if(metadata.getMessage().isNoteOn())
            m_AudioContainer.playAudio = true;
        else if(metadata.getMessage().isNoteOff()){
            stopAudio();
        }
    }

    // Check for generic play request.
    if(m_AudioContainer.playAudio){
        playAudio(buffer, totalNumOutputChannels, numSamples);
    }
}

//==============================================================================
bool AudioPluginAudioProcessor::hasEditor() const{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor(){
    return new AudioPluginAudioProcessorEditor (*this);
}

//==============================================================================
void AudioPluginAudioProcessor::getStateInformation(juce::MemoryBlock& destData){
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
    juce::ignoreUnused (destData);

    auto state = valueTreeState.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void AudioPluginAudioProcessor::setStateInformation(const void* data, int sizeInBytes){
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
    juce::ignoreUnused (data, sizeInBytes);

    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
 
    if (xmlState.get() != nullptr)
        if (xmlState->hasTagName (valueTreeState.state.getType()))
            valueTreeState.replaceState (juce::ValueTree::fromXml (*xmlState));
}

//------------------------------------------------------------//
// Audio playback methods.

void AudioPluginAudioProcessor::playAudio(juce::AudioBuffer<float>& buffer, int totalNumOutputChannels, int numSamples){
    buffer.clear();
    
    for (int channel = 0; channel < totalNumOutputChannels; ++channel){
        auto* channelData = buffer.getWritePointer(channel);
        juce::ignoreUnused(channelData);
        
        for(int sample = 0; sample < numSamples; ++sample){
            // Add samples to buffer if max length of samples is not exceeded.
            if(m_AudioContainer.sampleIndex[channel] < m_Generator.M_NUM_SAMPLES){
                channelData[sample] = m_AudioContainer.audio[m_AudioContainer.sampleIndex[channel]];
                m_AudioContainer.sampleIndex.set(channel, m_AudioContainer.sampleIndex[channel] + 1);
            }
            else{
                stopAudio();
            }
        }
    }
}

void AudioPluginAudioProcessor::stopAudio(){
    m_AudioContainer.sampleIndex.clear();
    m_AudioContainer.playAudio = false;
}

//==============================================================================
// This creates new instances of the plugin.
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter(){
    return new AudioPluginAudioProcessor();
}
