#include "Headers.h"


//==============================================================================
AudioPluginAudioProcessor::AudioPluginAudioProcessor()
    :   AudioProcessor(BusesProperties()
            #if ! JucePlugin_IsMidiEffect
                #if ! JucePlugin_IsSynth
                    .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                #endif
                .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
            #endif
        ),
        m_ValueTreeState(*this, nullptr, getName(), {}),
        m_Parameters(m_ValueTreeState.state){
    m_ValueTreeState.state.addListener(this);
}

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
void AudioPluginAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock){
    juce::ignoreUnused (sampleRate, samplesPerBlock);
}

void AudioPluginAudioProcessor::releaseResources(){}
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

void AudioPluginAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages){
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
bool AudioPluginAudioProcessor::hasEditor() const{return true;}

juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor(){return new AudioPluginAudioProcessorEditor (*this);}

//==============================================================================
void AudioPluginAudioProcessor::getStateInformation(juce::MemoryBlock& destData){
    if(m_Parameters.isInit){
        auto stateCopy = m_ValueTreeState.state.createCopy();
        m_Parameters.clearSamples(stateCopy);
        std::unique_ptr<juce::XmlElement> xml(stateCopy.createXml());
        copyXmlToBinary(*xml, destData);
    }
    else{m_Parameters.isInit = true;}
}

void AudioPluginAudioProcessor::setStateInformation(const void* data, int sizeInBytes){
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState.get() != nullptr){
        if (xmlState->hasTagName(m_ValueTreeState.state.getType())){
            m_ValueTreeState.state.copyPropertiesAndChildrenFrom(juce::ValueTree::fromXml(*xmlState), nullptr);
            rebuildState();
        }
    }
}

void AudioPluginAudioProcessor::rebuildState(){
    m_Parameters.rebuildSamples();
    m_Parameters.updateMap.setValue(true);
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

void AudioPluginAudioProcessor::playSample(){
    m_AudioContainer.sampleIndex.clear();
    m_AudioContainer.playAudio = true;
}

void AudioPluginAudioProcessor::addSample(juce::ValueTree node){
    m_AudioContainer.audio.clear();

    juce::Array<float> sample;
    sample.ensureStorageAllocated(Generator::M_NUM_SAMPLES);

    juce::Array<var>* values = node.getProperty(Parameters::sampleProp).getArray();
    for(int i = 0; i < Generator::M_NUM_SAMPLES; i++)
        sample.insert(i, (*values)[i]);

    m_AudioContainer.audio.addArray(sample);
}

//------------------------------------------------------------//
// Callback methods.

void AudioPluginAudioProcessor::valueTreePropertyChanged(juce::ValueTree& node, const juce::Identifier& id){
    if(id == Parameters::isActiveProp){
        if(node.getProperty(Parameters::isActiveProp)){
            addSample(node);
        }
    }
}

//==============================================================================
// This creates new instances of the plugin.
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter(){return new AudioPluginAudioProcessor();}
