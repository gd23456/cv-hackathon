import Foundation
import NaturalLanguage

public struct ParsedObject: Sendable, Equatable {
    public let noun: String
    /// The COCO class the detector can check, or nil when it can't see this object.
    public let label: String?
    /// Only set when the prompt states a number above one.
    public let count: Int?
    public let color: String?
}

public enum SpatialRelation: String, Sendable, Equatable {
    case leftOf = "left of"
    case rightOf = "right of"
    case above
    case below
}

public struct ParsedRelation: Sendable, Equatable {
    public let subject: Int
    public let relation: SpatialRelation
    public let object: Int
}

public struct ParsedPrompt: Sendable, Equatable {
    public let objects: [ParsedObject]
    public let relations: [ParsedRelation]

    public var namesColor: Bool { objects.contains { $0.color != nil } }

    /// Cheap estimate of how hard the prompt is: objects, plus every attribute
    /// and relation they must satisfy. Spatial relations are the hardest.
    public var complexity: Int {
        objects.count + relations.count * 2
            + objects.filter { $0.color != nil }.count
            + objects.filter { $0.count != nil }.count
    }
}

/// Turns a prompt into checkable facts using Apple's on-device NaturalLanguage tagger.
public enum PromptParser {
    public static func parse(_ prompt: String) -> ParsedPrompt {
        let text = prompt.lowercased()
        let tagger = NLTagger(tagSchemes: [.lexicalClass, .lemma])
        tagger.string = text

        struct Word { let text: String; let lemma: String; let isNoun: Bool }
        var words: [Word] = []
        let options: NLTagger.Options = [.omitWhitespace, .omitPunctuation]
        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .lexicalClass, options: options) { tag, range in
            let word = String(text[range])
            let lemma = tagger.tag(at: range.lowerBound, unit: .word, scheme: .lemma).0?.rawValue ?? word
            words.append(Word(text: word, lemma: lemma, isNoun: tag == .noun))
            return true
        }

        var objects: [ParsedObject] = []
        var relations: [ParsedRelation] = []
        var pendingCount: Int?
        var pendingColor: String?
        var between: [String] = []
        var index = 0

        func addObject(noun: String, label: String?) {
            let relation = relationIn(between)
            objects.append(ParsedObject(noun: noun, label: label, count: pendingCount, color: pendingColor))
            if let relation, objects.count >= 2 {
                relations.append(ParsedRelation(subject: objects.count - 2, relation: relation, object: objects.count - 1))
            }
            pendingCount = nil
            pendingColor = nil
            between = []
        }

        while index < words.count {
            let word = words[index]
            if let number = numberWords[word.text] ?? Int(word.text), number > 1 {
                pendingCount = number
            } else if let color = colorNames[word.text] {
                pendingColor = color
            } else if index + 1 < words.count, let label = cocoTwoWord["\(word.lemma) \(words[index + 1].lemma)"] {
                addObject(noun: "\(word.text) \(words[index + 1].text)", label: label)
                index += 1
            } else if let label = cocoLabel(word.lemma) ?? cocoLabel(singular(word.text)) {
                addObject(noun: word.text, label: label)
            } else if word.isNoun, !ignoredNouns.contains(word.lemma), !ignoredNouns.contains(word.text) {
                addObject(noun: word.text, label: nil)
            } else {
                between.append(word.text)
            }
            index += 1
        }
        return ParsedPrompt(objects: objects, relations: relations)
    }

    private static func relationIn(_ words: [String]) -> SpatialRelation? {
        let phrase = " " + words.joined(separator: " ") + " "
        if phrase.contains(" left ") { return .leftOf }
        if phrase.contains(" right ") { return .rightOf }
        if phrase.contains(" above ") || phrase.contains(" over ") || phrase.contains(" top ") { return .above }
        if phrase.contains(" below ") || phrase.contains(" under ") || phrase.contains(" beneath ") { return .below }
        return nil
    }

    private static func singular(_ word: String) -> String {
        word.hasSuffix("s") ? String(word.dropLast()) : word
    }

    static func cocoLabel(_ word: String) -> String? {
        if cocoOneWord.contains(word) { return word }
        return synonyms[word]
    }

    private static let numberWords = [
        "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    ]
    static let colorNames = [
        "red": "red", "orange": "orange", "yellow": "yellow", "green": "green", "blue": "blue",
        "purple": "purple", "violet": "purple", "pink": "pink", "brown": "brown", "black": "black",
        "white": "white", "gray": "gray", "grey": "gray",
    ]
    private static let ignoredNouns: Set<String> = [
        "left", "right", "top", "bottom", "front", "back", "side", "middle", "center", "background", "foreground",
        "picture", "photo", "photograph", "image", "painting", "drawing", "illustration", "render", "scene", "view",
        "style", "close-up", "closeup", "shot", "art", "day", "night", "morning", "evening", "detail", "quality", "light", "lighting",
    ]
    private static let cocoOneWord: Set<String> = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "kite", "skateboard", "surfboard", "bottle", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "pizza", "donut", "cake", "chair", "couch", "bed",
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "toothbrush",
    ]
    private static let cocoTwoWord = [
        "traffic light": "traffic light", "fire hydrant": "fire hydrant", "stop sign": "stop sign",
        "parking meter": "parking meter", "sports ball": "sports ball", "baseball bat": "baseball bat",
        "baseball glove": "baseball glove", "tennis racket": "tennis racket", "wine glass": "wine glass",
        "hot dog": "hot dog", "potted plant": "potted plant", "dining table": "dining table",
        "cell phone": "cell phone", "teddy bear": "teddy bear", "hair drier": "hair drier",
    ]
    private static let synonyms = [
        "man": "person", "woman": "person", "boy": "person", "girl": "person", "child": "person", "kid": "person",
        "people": "person", "lady": "person", "guy": "person", "puppy": "dog", "retriever": "dog", "labrador": "dog", "kitten": "cat", "bike": "bicycle",
        "motorbike": "motorcycle", "plane": "airplane", "aeroplane": "airplane", "automobile": "car", "mug": "cup",
        "sofa": "couch", "television": "tv", "phone": "cell phone", "table": "dining table", "plant": "potted plant",
        "ball": "sports ball", "fridge": "refrigerator", "bag": "handbag", "doughnut": "donut",
    ]
}
