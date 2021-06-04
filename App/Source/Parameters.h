#pragma once


struct Parameters{
    juce::ValueTree rootNode;

    // Type identifiers.
    static juce::Identifier sunType;
    static juce::Identifier planetType;

    // Property identifiers.
    static juce::Identifier idProp;
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
    void addPlanetNode(const juce::String&);
    void removePlanetNode(const juce::String&);

    // Tensor operations.
    void generateLatents(ValueTree);
    void generateSample(ValueTree, at::Tensor);
};