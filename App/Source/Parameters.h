#pragma once


struct Parameters{
    juce::ValueTree rootNode;

    // Type identifiers.
    juce::Identifier sunType;
    juce::Identifier planetType;

    // Property identifiers.
    juce::Identifier diameter;
    juce::Identifier posX;
    juce::Identifier posY;
    juce::Identifier posCentreX;
    juce::Identifier posCentreY;
    juce::Identifier colour;
    juce::Identifier latents;
    juce::Identifier lerpLatents;
    juce::Identifier sample;


    // Constructors and destructors.
    Parameters(juce::ValueTree);
    ~Parameters();

    // Restructuring methods.
    void addSunNode();
    void addPlanetNode();
};
