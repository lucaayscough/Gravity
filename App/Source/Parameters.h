#pragma once


struct Parameters: public juce::ValueTree::Listener{
    // Constructors and destructors.
    Parameters(juce::ValueTree);
    ~Parameters() override;

    // Restructuring methods.
    void addSunNode();
    void addPlanetNode();
    void removePlanetNode(const juce::String&);
    void clearSamples(juce::ValueTree);
    void rebuildSamples();

    // Tensor operations.
    void generateLatents(juce::ValueTree);
    void generateLerpLatents(juce::ValueTree);
    void generateSample(juce::ValueTree, at::Tensor);
    void generateOldSample(juce::ValueTree);
    void generateNewSample(juce::ValueTree);
    void mixLatents();

    // Get methods.
    juce::ValueTree getSunNode();
    juce::ValueTree getRootPlanetNode();
    std::int64_t getSeed(juce::ValueTree);
    at::Tensor getLatents(juce::ValueTree, juce::Identifier&);
    juce::ValueTree getActivePlanet();
    juce::String getID(juce::ValueTree);
    float getDistance(juce::ValueTree, juce::ValueTree);
    float getForceVector(juce::ValueTree, juce::ValueTree);

    // Set methods.
    void setActivePlanet(juce::ValueTree);
    void setRandomID(juce::ValueTree);
    void setRandomSeed(juce::ValueTree);
    void setLatents(juce::ValueTree, juce::Identifier&, at::Tensor&);

    // Callback methods.
    void valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier&) override;

    // Member variables.
    juce::ValueTree rootNode;
    bool isInit = false;
    juce::Value updateMap;
    const juce::String SUN_ID = "Sun";

    // Type identifiers.
    static juce::Identifier sunType;
    static juce::Identifier rootPlanetType;
    static juce::Identifier planetType;

    // Property identifiers.
    static juce::Identifier idProp;
    static juce::Identifier isActiveProp;
    static juce::Identifier diameterProp;
    static juce::Identifier posXProp;
    static juce::Identifier posYProp;
    static juce::Identifier posCentreXProp;
    static juce::Identifier posCentreYProp;
    static juce::Identifier colourProp;
    static juce::Identifier seedProp;
    static juce::Identifier latentsProp;
    static juce::Identifier lerpLatentsProp;
    static juce::Identifier sampleProp;

    // Callback signalers.
    static juce::Identifier updateGraphSignal;
    static juce::Identifier generateSampleSignal;

};
