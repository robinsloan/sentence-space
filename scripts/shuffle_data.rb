lines = []

File.open(ARGV[0],"r").each_line do |line|
  lines << line
end

puts "Shuffling..."

lines.shuffle!

output = File.open("shuffled.txt", "w")

puts "Now writing lines..."

lines.each do |line|
  output << line
end

output.close
puts "Done!"
