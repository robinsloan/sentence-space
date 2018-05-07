require "cgi"

unless (ARGV[0] && ARGV[1])
  puts "You need to specify two sentences."
  exit
end

sentence_origin = CGI.escape(ARGV[0])
sentence_dest = CGI.escape(ARGV[1])

resp = `curl "http://0.0.0.0:5099/gradient?s1=#{sentence_origin}&s2=#{sentence_dest}"`
#resp = `curl "http://0.0.0.0:5099/neighborhood?s1=#{sentence_origin}&mag=0.2"`

puts resp

puts "Done!"
