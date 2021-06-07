#pragma once


struct Parameters{
    juce::ValueTree rootNode;

    // Type identifiers.
    static juce::Identifier sunType;
    static juce::Identifier planetType;

    // Property identifiers.
    static juce::Identifier idProp;
    static juce::Identifier mapWidthProp;
    static juce::Identifier mapHeightProp;
    static juce::Identifier diameterProp;
    static juce::Identifier posXProp;
    static juce::Identifier posYProp;
    static juce::Identifier posCentreXProp;
    static juce::Identifier posCentreYProp;
    static juce::Identifier colourProp;
    static juce::Identifier latentsProp;
    static juce::Identifier lerpLatentsProp;
    static juce::Identifier sampleProp;

    // Constructors and destructors.
    Parameters(juce::ValueTree);
    ~Parameters();

    // Restructuring methods.
    void addSunNode();
    void addPlanetNode();
    void removePlanetNode(const juce::String&);

    // Tensor operations.
    void generateLatents(juce::ValueTree);
    void generateLerpLatents(juce::ValueTree);
    void generateSample(juce::ValueTree, at::Tensor);
    void mixLatents();

    // Get methods.
    at::Tensor getLatents(juce::ValueTree, juce::Identifier&);
    juce::String getID(juce::ValueTree);
    float getDistance(juce::ValueTree, juce::ValueTree);
    float getForceVector(juce::ValueTree, juce::ValueTree);

    // Set methods.
    void setLatents(juce::ValueTree, juce::Identifier&, at::Tensor&);
};
