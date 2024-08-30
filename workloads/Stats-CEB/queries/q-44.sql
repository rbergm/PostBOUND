SELECT COUNT(*)
FROM postHistory AS ph,
  posts AS p,
  users AS u,
  badges AS b
WHERE u.Id = p.OwnerUserId
  AND p.OwnerUserId = ph.UserId
  AND ph.UserId = b.UserId
  AND ph.CreationDate >= CAST('2010-09-06 11:41:43' AS timestamp)
  AND ph.CreationDate <= CAST('2014-09-03 16:41:18' AS timestamp)
  AND p.Score >= -1
  AND p.ViewCount >= 0
  AND p.ViewCount <= 39097
  AND p.AnswerCount >= 0
  AND p.CommentCount >= 0
  AND p.CommentCount <= 11
  AND p.FavoriteCount <= 10
  AND p.CreationDate >= CAST('2010-08-13 02:18:09' AS timestamp)
  AND p.CreationDate <= CAST('2014-09-09 10:20:27' AS timestamp)
  AND u.Views >= 0
  AND u.DownVotes >= 0
  AND u.DownVotes <= 0
  AND u.UpVotes >= 0
  AND u.UpVotes <= 37;
