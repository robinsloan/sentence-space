require "cgi"

unless (ARGV[0] && ARGV[1])
  puts "You need to specify two sentences."
  exit
end

sentence_origin = CGI.escape(ARGV[0])
sentence_dest = CGI.escape(ARGV[1])

resp = `curl "http://0.0.0.0:5000/interpolate?s1=#{sentence_origin}&s2=#{sentence_dest}"`

puts resp.gsub(/\\u2581/," ")

puts "Done!"