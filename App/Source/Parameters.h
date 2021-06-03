#pragma once


struct Parameters{
    juce::ValueTree rootNode;

    // Type identifiers.
    juce::Identifier sunType;
    juce::Identifier planetType;

    // Constructors and destructors.
    Parameters(juce::ValueTree);
    ~Parameters();

    // Structuring methods.
    void addSunNode();
};
